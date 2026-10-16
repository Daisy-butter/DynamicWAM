"""Backend-independent empty-row attention, including finite zero gradients."""
from unittest.mock import patch
import unittest
import torch
from torch.nn import functional as F
from dynamicwam.models.motion_kv_conditioner import MotionKVConditioner

class EmptyAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261016);torch.set_num_threads(2)
        self.model=MotionKVConditioner(dim=24,layers=2,heads=2,head_dim=12,text_dim=16,
            world_heads=2,world_head_dim=6,world_dim=16,kv_rank=3)
        self.q=torch.randn(2,3,16,requires_grad=True)
        self.k=torch.randn(2,5,16,requires_grad=True)
        self.v=torch.randn(2,5,16,requires_grad=True)
    def raw(self,q,k,v,valid=None):
        b=q.shape[0]
        shape=lambda x:x.reshape(b,-1,2,8).transpose(1,2)
        result=F.scaled_dot_product_attention(shape(q),shape(k),shape(v),attn_mask=None if valid is None else valid[:,None,None,:])
        return result.transpose(1,2).reshape(b,-1,16)
    def test_nonempty_values_and_gradients_are_unchanged(self):
        valid=torch.tensor([[True,False,True,False,False],[False,True,False,False,True]])
        before=self.raw(self.q,self.k,self.v,valid)
        after=self.model._attention(self.q,self.k,self.v,valid)
        self.assertTrue(torch.equal(before,after))
        grads0=torch.autograd.grad(before.square().sum(),(self.q,self.k,self.v),retain_graph=True)
        grads1=torch.autograd.grad(after.square().sum(),(self.q,self.k,self.v))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(grads0,grads1)))
    def test_empty_rows_are_made_finite_before_backend_and_have_zero_gradients(self):
        valid=torch.tensor([[True,False,True,False,False],[False,False,False,False,False]])
        q=self.q.detach().clone();k=self.k.detach().clone();v=self.v.detach().clone()
        q[1]=float('nan');k[1]=float('nan');v[1]=float('nan')
        q.requires_grad_();k.requires_grad_();v.requires_grad_()
        original=F.scaled_dot_product_attention;calls=[]
        def backend(q,k,v,*,attn_mask):
            self.assertTrue(bool(attn_mask.any(-1).all()))
            self.assertTrue(bool(torch.isfinite(q).all() & torch.isfinite(k).all() & torch.isfinite(v).all()))
            calls.append(True)
            return original(q,k,v,attn_mask=attn_mask)
        with patch('dynamicwam.models.motion_kv_conditioner.F.scaled_dot_product_attention',backend):
            result=self.model._attention(q,k,v,valid)
        self.assertEqual(len(calls),1);self.assertTrue(result[1].eq(0).all())
        result.square().sum().backward()
        for x in (q,k,v):
            self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(x.grad[1].eq(0).all())
        expected=self.raw(self.q[:1],self.k[:1],self.v[:1],valid[:1])
        torch.testing.assert_close(result[:1],expected,atol=0,rtol=0)
    def test_all_empty_rows_zero_for_fp32_and_bf16(self):
        for dtype in (torch.float32,torch.bfloat16):
            q=self.q.detach().to(dtype).requires_grad_();k=self.k.detach().to(dtype).requires_grad_();v=self.v.detach().to(dtype).requires_grad_()
            result=self.model._attention(q,k,v,torch.zeros(2,5,dtype=torch.bool))
            self.assertTrue(result.eq(0).all());result.sum().backward()
            for x in (q,k,v):self.assertTrue(x.grad.eq(0).all())
    def test_unmasked_path_is_bit_exact(self):
        self.assertTrue(torch.equal(self.model._attention(self.q,self.k,self.v),self.raw(self.q,self.k,self.v)))

if __name__=='__main__':unittest.main()
