# 三轮审稿式迭代记录

完整初始接线后，Round0 基础31项CPU通过。以下是实际审查与修复，不把连续跑同一测试称作新迭代。

## Round1：机制与训练数据

发现：专家替代式监督偏离历史实验主教师；仅age0 bootstrap无法训练受保护fresh行；丢弃全部越界目标会饿死局部反馈；按scene/seed分层使半量实际保留7467集。
修改：保留当前448/6/7分频主干，改为同命令配对噪声的fresh Full主教师、方向一致专家辅助；未裁剪endpoint Huber给有界输出方向梯度；开放原始所有年龄窗口；task/style整集半量。
验证：33项CPU通过，包含越界目标仍有正确方向梯度、半量整集统计、上下文/计划P连接、夹爪/fresh保留和历史V9/Graph回归。真实半量样本检查另存SSD诊断回执。GPU未运行。

## Round2

发现并修复：交互向量缺少可重验的真实时间与命令身份；部署previous proposal保存了边界处理之前的版本；回放可能悄悄补零而接受旧schema；专家教师噪声在不同episode中完全相同；训练入口需要复核真实query exposure和已选首waypoint。

新增FP64 interval/int64 command id/连续suffix校验；actual qpos/qvel按配置joint顺序读取；previous proposal绑定实际边界处理后的目标；V10回放schema拒绝旧缺失数据；episode绑定教师底噪；source/current clamping与exposure检查。8条命令历史只包含已经完成的TOPP动作。

验证：71项CPU通过，覆盖真实关节读取、未来/错误命令历史拒绝、native history、Full噪声对齐、优化器/checkpoint/分布式配置回归。首次宽回归暴露旧V7测试把当前架构误当作deployment-only补丁的假设；测试改用实际冻结V7源码构造正例，同时验证当前V10和任何训练模型修改仍被V7兼容门拒绝，生产门禁没有放宽。原失败日志保留。GPU仍未运行。

## Round3

发现并修复：专家训练的 executed_deviation 固定为零，部署却读取实际发出命令相对源计划的均值偏差；现在专家按同一命令前缀计算，回放保留实测值。future_events 必须严格等于 query command +1/+2/+4/+8，拒绝错位标签。该偏差不是实际关节误差，后者仍由独立 interaction 提供。

启动审查还发现 receipt 哈希函数只接受 Path、CLI 传入 str 的错误，故障注入首次失败后已修复。最终预约入口绑定干净源码、配置、数据索引、Full profile 和三轮证据；CPU 记录明确 formal_training_ready=false。GPU现场空闲、真实Graph、新M/原生MP4、三阶段八卡短训及严格恢复必须另外全部通过。

数据采集现在按半量元数据预先固定35个 clean/train 场景，不从评测seed或策略成功结果选择；若DOMINO准备不接受指定seed，直接失败。历史V9 collection的专家轨迹数值失败保留，不宣称已修复仿真器，也不自动重试。正式700评测不在本次定时训练范围内。

验证：最终100项CPU通过（49.42s），包括两进程CPU分布式/全状态恢复回归与启动故障注入，不冒充八卡验收。生产宽度448/6/7的独立CPU三次更新、模型/优化器严格恢复及实际数据读取另存SSD回执。新模型的完整GPU、真实TOPP、八卡训练就绪与成功率均仍待验收。

审稿结论：机制和工程路径足以进入有门禁的验证阶段；不能由历史+11成功推导本版SR保底或SOTA。专家配对窗口为两命令间隔，实际回放/部署为一命令；专家没有真实joint历史，使用缺失mask。必须通过真实回放补足控制器信息，并在训练后做同协议零残差和完整模型配对实验。初始Full策略回放不等于训练后学生分布覆盖。
