"""Immutable, complete CPU-PhysX scene snapshots for an independent renderer.

Only the owner of a scene may call this class.  ``apply`` never steps physics or
renders; the receiving process calls ``scene.update_render`` and reads cameras
afterwards.  Policy proprioception must remain the source packet's drive targets,
not the renderer copy's robot API.  Meshes, materials and projection are immutable
for an episode; unsupported visual components fail closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = 'dynamicwam_domino_scene_snapshot'
VERSION = 1


class UnsupportedSnapshotScene(RuntimeError):
    pass


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array(value: Any) -> list:
    array = np.asarray(value)
    if array.dtype.kind in 'fc' and not np.isfinite(array).all():
        raise ValueError('nonfinite scene snapshot array')
    return array.tolist()


def _array_hash(value: Any) -> dict:
    array = np.ascontiguousarray(value)
    if array.dtype.kind in 'fc' and not np.isfinite(array).all():
        raise ValueError('nonfinite visual geometry')
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _pose(value: Any) -> list[float]:
    result = np.concatenate((np.asarray(value.p), np.asarray(value.q))).astype(np.float64)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError('invalid pose')
    if abs(float(np.linalg.norm(result[3:]))-1.) > 1e-4:
        raise ValueError('pose quaternion is not normalized')
    return result.tolist()


def _pose_object(value: Any) -> Any:
    import sapien
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError('invalid incoming pose')
    if abs(float(np.linalg.norm(array[3:]))-1.) > 1e-4:
        raise ValueError('invalid incoming quaternion')
    return sapien.Pose(array[:3], array[3:])


def _enabled(component: Any) -> bool:
    return bool(component.is_enabled)


def _set_enabled(component: Any, value: bool) -> None:
    if bool(value) != _enabled(component):
        (component.enable if value else component.disable)()


def _shape_state(shape: Any) -> dict:
    return dict(local_pose=_pose(shape.local_pose),
                scale=_array(shape.scale) if hasattr(shape, 'scale') else None,
                front_face=str(shape.front_face))


def _values(owner: Any, names: tuple[str, ...]) -> dict:
    result = {}
    for name in names:
        if hasattr(owner, name):
            value = getattr(owner, name)
            if isinstance(value, (str, bool, int, float)):
                result[name] = value
            else:
                result[name] = _array(value)
    return result


_MATERIAL_FIELDS = ('base_color', 'emission', 'roughness', 'metallic', 'specular',
                    'transmission', 'transmission_roughness', 'ior')
_TEXTURE_FIELDS = ('base_color_texture', 'emission_texture', 'normal_texture',
                  'roughness_texture', 'metallic_texture', 'transmission_texture')
_LIGHT_FIELDS = ('color', 'shadow', 'shadow_near', 'shadow_far', 'shadow_map_size',
                 'shadow_half_size', 'inner_fov', 'outer_fov')
_SHAPES = {'RenderShapeTriangleMesh', 'RenderShapeBox', 'RenderShapePlane',
           'RenderShapeSphere', 'RenderShapeCylinder', 'RenderShapeCapsule'}
_LIGHTS = {'RenderPointLightComponent', 'RenderDirectionalLightComponent',
           'RenderSpotLightComponent'}


class SceneSnapshotBridge:
    """Map a separately prepared identical scene; no shared SAPIEN objects.

    Identity uses insertion-ordered entities, geometry/content fingerprints and
    joint topology. Names alone and unordered ``get_all_actors`` are insufficient.
    CPU FK restore is checked against every source link pose on every snapshot.
    """

    def __init__(self, env: Any, *, pose_tolerance: float = 1e-4) -> None:
        self.env = env
        self.scene = env.scene
        self.pose_tolerance = float(pose_tolerance)
        if not 0 < self.pose_tolerance <= 1e-3:
            raise ValueError('pose tolerance must be in (0, 1e-3]')
        if 'gpu' in type(self.scene.physx_system).__name__.lower():
            raise UnsupportedSnapshotScene('only CPU PhysX restore is supported')
        if bool(getattr(env, 'crazy_random_light', False)):
            raise UnsupportedSnapshotScene('crazy_random_light requires an explicit RNG/light protocol')
        if self.scene.render_system.cubemap is not None:
            raise UnsupportedSnapshotScene('cubemap content identity is not implemented')
        self.entities = list(self.scene.entities)
        self._entity_ids = [int(e.per_scene_id) for e in self.entities]
        if len(set(self._entity_ids)) != len(self._entity_ids):
            raise ValueError('duplicate scene entity IDs')
        self._index = {eid: i for i, eid in enumerate(self._entity_ids)}
        self._textures: list[tuple[Any, dict]] = []
        self._initializing = True
        self._asset_hashes: dict[str, str] = {}
        self.bodies = []
        self.cameras = []
        self.lights = []
        self._parts = []
        self._shapes = []
        self._component_signature = []
        self._component_refs = []
        manifests = []
        for ei, entity in enumerate(self.entities):
            components = list(entity.components)
            self._component_refs.append(tuple(components))
            self._component_signature.append(tuple(type(c).__name__ for c in components))
            manifest = dict(index=ei, name=str(entity.name), components=[])
            for ci, component in enumerate(components):
                kind = type(component).__name__
                item = dict(type=kind, name=str(component.name))
                key = [ei, ci]
                if kind == 'RenderBodyComponent':
                    shapes = list(component.render_shapes)
                    item['shapes'] = [self._shape_identity(s) for s in shapes]
                    self.bodies.append((key, component, shapes))
                elif kind == 'RenderCameraComponent':
                    item['projection'] = self._camera_projection(component)
                    item['local_pose'] = _pose(component.local_pose)
                    self.cameras.append((key, component))
                elif kind in _LIGHTS:
                    item['properties'] = _values(component, _LIGHT_FIELDS)
                    item['local_pose'] = _pose(component.local_pose)
                    self.lights.append((key, component))
                elif not kind.startswith('Physx'):
                    raise UnsupportedSnapshotScene(f'unsupported scene component: {kind}')
                manifest['components'].append(item)
            manifests.append(manifest)
        self.articulations = sorted(list(self.scene.get_all_articulations()),
                                    key=lambda a: self._entity_index(a.get_links()[0].entity))
        self.link_indices = set()
        articulation_manifest = []
        for articulation in self.articulations:
            links = list(articulation.get_links())
            indices = [self._entity_index(link.entity) for link in links]
            if self.link_indices.intersection(indices):
                raise ValueError('link belongs to multiple articulations')
            self.link_indices.update(indices)
            joints = list(articulation.get_joints())
            articulation_manifest.append(dict(links=indices,
                link_names=[str(link.entity.name) for link in links],
                joints=[dict(name=str(j.name), type=str(j.type), dof=int(j.dof)) for j in joints],
                qpos_size=int(np.asarray(articulation.get_qpos()).size)))
        self.nonlink_indices = [i for i in range(len(self.entities)) if i not in self.link_indices]
        self._static_materials = self._material_signatures()
        self._static_shape_states = [[_shape_state(s) for s in shapes]
                                     for _, _, shapes in self.bodies]
        self._projection_identity = [self._camera_projection(c) for _, c in self.cameras]
        import sapien
        render_settings = dict(shader=Path(sapien.render.get_camera_shader_dir()).name,
            samples_per_pixel=int(sapien.render.get_ray_tracing_samples_per_pixel()),
            path_depth=int(sapien.render.get_ray_tracing_path_depth()),
            denoiser=str(sapien.render.get_ray_tracing_denoiser()))
        if render_settings != dict(shader='rt', samples_per_pixel=32, path_depth=8, denoiser='oidn'):
            raise UnsupportedSnapshotScene(f'expected unchanged DOMINO RT rendering: {render_settings}')
        self.identity = dict(format=FORMAT+'_identity', version=VERSION,
            sapien_version=str(sapien.__version__), entities=manifests,
            articulations=articulation_manifest,
            nonlink_indices=self.nonlink_indices,
            renderer=render_settings,
            constraints=dict(physics='cpu', renderer_never_steps=True,
                immutable_geometry=True, immutable_materials=True, immutable_projection=True,
                no_crazy_random_light=True, no_cubemap=True))
        self.identity['sha256'] = _json_hash(self.identity)
        self._initializing = False
        self._sequence = 0

    def _entity_index(self, entity: Any) -> int:
        try:
            return self._index[int(entity.per_scene_id)]
        except KeyError as exc:
            raise ValueError('render/articulation entity is outside scene') from exc

    def _texture_identity(self, texture: Any) -> dict | None:
        if texture is None:
            return None
        for old, descriptor in self._textures:
            if texture == old:
                return descriptor
        if not self._initializing:
            raise UnsupportedSnapshotScene('episode texture replacement is unsupported')
        if type(texture).__name__ != 'RenderTexture2D':
            raise UnsupportedSnapshotScene(f'unsupported texture {type(texture).__name__}')
        descriptor = dict(type=type(texture).__name__, filename=str(texture.filename),
            width=int(texture.width), height=int(texture.height), channels=int(texture.channels),
            format=str(texture.format), filter_mode=str(texture.filter_mode),
            address_mode=str(texture.address_mode), mipmap_levels=int(texture.mipmap_levels),
            is_srgb=bool(texture.is_srgb))
        filename = descriptor['filename']
        if filename and Path(filename).is_file():
            if filename not in self._asset_hashes:
                self._asset_hashes[filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            descriptor['file_sha256'] = self._asset_hashes[filename]
        else:
            # One-time initialization only; never download textures in capture().
            descriptor['pixels'] = _array_hash(texture.download())
        self._textures.append((texture, descriptor))
        return descriptor

    def _material_identity(self, material: Any) -> dict:
        if material is None:
            raise UnsupportedSnapshotScene('missing render material')
        return dict(properties=_values(material, _MATERIAL_FIELDS),
            textures={name: self._texture_identity(getattr(material, name)) for name in _TEXTURE_FIELDS})

    def _shape_identity(self, shape: Any) -> dict:
        kind = type(shape).__name__
        if kind not in _SHAPES:
            raise UnsupportedSnapshotScene(f'unsupported render shape: {kind}')
        result = dict(type=kind, name=str(shape.name), local_pose=_pose(shape.local_pose),
            front_face=str(shape.front_face),
            geometry=_values(shape, ('half_size', 'radius', 'half_length', 'scale')),
            filename=str(getattr(shape, 'filename', '')), parts=[])
        self._shapes.append(shape)
        for part in shape.parts:
            self._parts.append(part)
            result['parts'].append(dict(vertices=_array_hash(part.vertices),
                triangles=_array_hash(part.triangles),
                normals=_array_hash(part.get_vertex_normal()), uv=_array_hash(part.get_vertex_uv()),
                material=self._material_identity(part.material)))
        if not result['parts']:
            raise UnsupportedSnapshotScene('render shape has no fingerprintable mesh parts')
        return result

    def _material_signatures(self) -> list[dict]:
        return [self._material_identity(part.material) for part in self._parts]

    @staticmethod
    def _camera_projection(camera: Any) -> dict:
        return dict(width=int(camera.width), height=int(camera.height),
            intrinsic=_array(camera.get_intrinsic_matrix()),
            projection=_array(camera.get_projection_matrix()))

    def _check_layout(self) -> None:
        entities = list(self.scene.entities)
        if [int(e.per_scene_id) for e in entities] != self._entity_ids:
            raise UnsupportedSnapshotScene('scene entities added, removed or reordered')
        components = [tuple(e.components) for e in entities]
        if [tuple(type(c).__name__ for c in cs) for cs in components] != self._component_signature:
            raise UnsupportedSnapshotScene('scene component layout changed')
        if any(any(now is not old for now, old in zip(current, expected))
               for current, expected in zip(components, self._component_refs)):
            raise UnsupportedSnapshotScene('scene component replaced during episode')
        for _, body, shapes in self.bodies:
            current_shapes = list(body.render_shapes)
            if len(current_shapes) != len(shapes) or any(now is not old for now, old in zip(current_shapes, shapes)):
                raise UnsupportedSnapshotScene('render shape layout changed')
        if [[_shape_state(s) for s in shapes] for _, _, shapes in self.bodies] != self._static_shape_states:
            raise UnsupportedSnapshotScene('attached render shape pose/scale/front-face mutation is unsupported')
        if self._material_signatures() != self._static_materials:
            raise UnsupportedSnapshotScene('episode material/texture mutation is unsupported')
        if [self._camera_projection(c) for _, c in self.cameras] != self._projection_identity:
            raise UnsupportedSnapshotScene('episode camera projection mutation is unsupported')

    def _refresh_wrist_poses(self) -> None:
        cameras = self.env.cameras
        if bool(getattr(cameras, 'collect_wrist_camera', False)):
            cameras.update_wrist_camera(self.env.robot.left_camera.get_pose(),
                                        self.env.robot.right_camera.get_pose())

    def capture(self) -> dict:
        """Read one physical instant without rendering or advancing physics."""
        self._check_layout()
        self._refresh_wrist_poses()
        self._sequence += 1
        clock = getattr(self.env, '_scene_step_clock', None)
        stamp = None if clock is None else float(clock.snapshot().time_seconds)
        if stamp is not None and (not np.isfinite(stamp) or stamp < 0):
            raise ValueError('invalid scene source time')
        return dict(format=FORMAT, version=VERSION, identity_sha256=self.identity['sha256'],
            sequence=self._sequence, source_sim_time_seconds=stamp,
            entities=[dict(index=i, pose=_pose(self.entities[i].pose)) for i in self.nonlink_indices],
            articulations=[dict(root_pose=_pose(a.get_root_pose()), qpos=_array(a.get_qpos()),
                link_poses=[_pose(link.entity.pose) for link in a.get_links()]) for a in self.articulations],
            bodies=[dict(key=key, enabled=_enabled(body), visibility=float(body.visibility),
                shading_mode=int(body.shading_mode), shapes=[dict(local_pose=_pose(s.local_pose),
                    scale=_array(s.scale) if hasattr(s, 'scale') else None,
                    front_face=str(s.front_face)) for s in shapes]) for key, body, shapes in self.bodies],
            cameras=[dict(key=key, local_pose=_pose(c.local_pose), global_pose=_pose(c.global_pose),
                enabled=_enabled(c)) for key, c in self.cameras],
            lights=[dict(key=key, local_pose=_pose(light.local_pose), enabled=_enabled(light),
                properties=_values(light, _LIGHT_FIELDS)) for key, light in self.lights],
            ambient_light=_array(self.scene.render_system.ambient_light))

    def apply(self, snapshot: dict) -> dict:
        """Restore in renderer owner, check FK, return oracle; never step/render."""
        self._check_layout()
        _json_hash(snapshot)  # Reject nonfinite/malformed numerical JSON before mutation.
        if (snapshot.get('format') != FORMAT or snapshot.get('version') != VERSION or
                snapshot.get('identity_sha256') != self.identity['sha256']):
            raise ValueError('renderer snapshot identity mismatch')
        if [v['index'] for v in snapshot['entities']] != self.nonlink_indices:
            raise ValueError('incomplete or reordered entity snapshot')
        if len(snapshot['articulations']) != len(self.articulations):
            raise ValueError('incomplete articulation snapshot')
        for field, expected in (('bodies', self.bodies), ('cameras', self.cameras), ('lights', self.lights)):
            if [v['key'] for v in snapshot[field]] != [row[0] for row in expected]:
                raise ValueError(f'incomplete {field} snapshot')
        # SAPIEN 3 forbids attached shape pose/scale setters, even for the same
        # value. They are part of the immutable scene identity; validate before
        # changing any dynamic pose, rather than attempting to restore them.
        if [item['shapes'] for item in snapshot['bodies']] != self._static_shape_states:
            raise UnsupportedSnapshotScene('incoming attached shape pose/scale/front-face differs')
        for articulation, value in zip(self.articulations, snapshot['articulations']):
            qpos = np.asarray(value['qpos'], dtype=np.float32)
            if qpos.shape != np.asarray(articulation.get_qpos()).shape or not np.isfinite(qpos).all():
                raise ValueError('invalid articulation qpos snapshot')
            if len(value['link_poses']) != len(articulation.get_links()):
                raise ValueError('incomplete link pose oracle')
            articulation.set_root_pose(_pose_object(value['root_pose']))
            articulation.set_qpos(qpos)
        for item in snapshot['entities']:
            self.entities[item['index']].set_pose(_pose_object(item['pose']))
        for (_, body, shapes), item in zip(self.bodies, snapshot['bodies']):
            if len(item['shapes']) != len(shapes):
                raise ValueError('incomplete shape snapshot')
            _set_enabled(body, item['enabled'])
            body.visibility = float(item['visibility'])
            body.shading_mode = int(item['shading_mode'])
        for (_, camera), item in zip(self.cameras, snapshot['cameras']):
            camera.set_local_pose(_pose_object(item['local_pose']))
            _set_enabled(camera, item['enabled'])
        for (_, light), item in zip(self.lights, snapshot['lights']):
            light.set_local_pose(_pose_object(item['local_pose']))
            _set_enabled(light, item['enabled'])
            for name, value in item['properties'].items():
                if name == 'shadow':
                    (light.enable_shadow if value else light.disable_shadow)()
                elif name == 'shadow_map_size':
                    if int(light.shadow_map_size) != int(value):
                        raise UnsupportedSnapshotScene('shadow map size changed')
                else:
                    setattr(light, name, value)
        self.scene.render_system.ambient_light = snapshot['ambient_light']
        comparisons = [(self.entities[v['index']].pose, v['pose']) for v in snapshot['entities']]
        for articulation, item in zip(self.articulations, snapshot['articulations']):
            comparisons.extend((link.entity.pose, expected) for link, expected
                               in zip(articulation.get_links(), item['link_poses']))
        comparisons.extend((c.global_pose, item['global_pose'])
                           for (_, c), item in zip(self.cameras, snapshot['cameras']))
        translation = rotation = 0.
        for actual, expected in comparisons:
            a = np.asarray(_pose(actual), dtype=np.float64)
            b = np.asarray(expected, dtype=np.float64)
            translation = max(translation, float(np.linalg.norm(a[:3]-b[:3])))
            rotation = max(rotation, float(min(np.linalg.norm(a[3:]-b[3:]), np.linalg.norm(a[3:]+b[3:]))))
        if max(translation, rotation) > self.pose_tolerance:
            raise ValueError(f'renderer restore pose oracle failed: translation={translation}, quaternion={rotation}')
        return dict(passed=True, identity_sha256=self.identity['sha256'], sequence=snapshot['sequence'],
            source_sim_time_seconds=snapshot['source_sim_time_seconds'],
            entity_count=len(self.entities), articulation_count=len(self.articulations),
            link_count=len(self.link_indices), camera_count=len(self.cameras),
            max_translation_error=translation, max_sign_invariant_quaternion_error=rotation,
            tolerance=self.pose_tolerance, physics_steps=0)
