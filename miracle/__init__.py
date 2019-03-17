from miracle.miracle_graph import MiracleGraph

graph = MiracleGraph()

create_variable = graph.create_variable

create_compression_graph = graph.create_compression_graph

assign_session = graph.assign_session

pretrain = graph.pretrain
train = graph.train
run_pretrain_op = graph.run_pretrain_op
run_train_op = graph.run_train_op

compress = graph.compress
load = graph.load