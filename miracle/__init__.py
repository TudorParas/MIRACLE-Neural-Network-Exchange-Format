from miracle.miracle_graph import MiracleGraph

graph = MiracleGraph()

create_variable = graph.create_variable

create_compression_graph = graph.create_compression_graph

assign_session = graph.assign_session
pretrain = graph.pretrain
train = graph.train
compress = graph.compress
load = graph.load