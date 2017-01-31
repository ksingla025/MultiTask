# MultiTask
Multi-task joint learning

from multitask import *
 
#### joint attention
trainer = MultiTask(num_steps=1400001, task1_start=20000, attention='true',joint='true',logs_path='./tmp/tensorflow_logs/joint_attention')
trainer.fit()

#### sequence attention
trainer = MultiTask(num_steps=2000001, task1_start=600000, attention='true',joint='false',logs_path='./tmp/tensorflow_logs/sequence_attention')
trainer.fit()

#### joint no attention
trainer = MultiTask(num_steps=1400001, task1_start=20000, attention='false',joint='true',logs_path='./tmp/tensorflow_logs/joint_noattention')
trainer.fit()

#### sequence no attention
trainer = MultiTask(num_steps=2000001, task1_start=600000, attention='false',joint='false',logs_path='./tmp/tensorflow_logs/sequence_noattention')
trainer.fit()

####some other parameters one can set

