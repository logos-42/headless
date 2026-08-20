"""headless 冒烟测试: LM1 最小规模跑通验证 (1 round, 2 iters, 不触发 migrate)"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
sys.path.insert(0, os.path.dirname(__file__))

from run_lm1_production import LM1System

t0 = time.time()
sys = LM1System(d_model=64, d_state=8, n_layers=1, iters_per_round=2, n_prop=1)
print("[smoke] model init OK, params=%.2fM (%.1fs)" % (
    sys._n_params() / 1e6, time.time() - t0))

t1 = time.time()
loss, t_train = sys.learn()
print("[smoke] learn OK loss=%.4f (%.1fs)" % (loss, t_train))

t2 = time.time()
rec = sys.evaluate(with_migrate=False)
print("[smoke] evaluate OK unseen=%.4f struct=%s seen=%.4f (%.1fs)" % (
    rec['unseen_mean'], rec['unseen_by_struct'], rec['seen_mean'],
    time.time() - t2))

sys.save('smoke')
print("[smoke] save OK -> checkpoints/lm1_smoke.pt")
print("[smoke] ALL PASSED (total %.1fs)" % (time.time() - t0))