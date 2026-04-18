import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aman/cs671_7/rl_eval_ws/install/eval_navigation'
