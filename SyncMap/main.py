from sklearn.metrics import normalized_mutual_info_score
#  problems
from OverlapChunkTest1 import *
from OverlapChunkTest2 import *
import sys

#  neurons
from SyncMap import *

save_dir = "output_files/"

arg_size = len(sys.argv)
if arg_size > 1:
	problem_type = sys.argv[1]
	save_filename = save_dir + sys.argv[2]
	save_truth_filename = save_dir + sys.argv[2] + "_truth"
else:
	save_filename = None
	problem_type = None


time_delay = 10

problem_type = int(problem_type)

if problem_type == 6:
	env = OverlapChunkTest1(time_delay)
	print("problem: OverlapChunkTest1")
if problem_type == 7:
	env = OverlapChunkTest2(time_delay)
	print("problem: OverlapChunkTest2")

output_size = env.getOutputSize()


print("Output Size: ", output_size)


sequence_length = 100000

# ###### SyncMap #####
number_of_nodes = output_size
adaptation_rate = 0.001 * output_size
print("Adaptation rate:", adaptation_rate)
map_dimensions = 2
neuron_group = SyncMap(number_of_nodes, map_dimensions, adaptation_rate)
# ###### SyncMap #####

input_sequence, input_class = env.getSequence(sequence_length)

neuron_group.input(input_sequence)
labels = neuron_group.organize()

print("Learned Labels: ", labels)
print("Correct Labels: ", env.trueLabel())
nmi_score = normalized_mutual_info_score(env.trueLabel(), labels)
print("NMI Score: ", nmi_score)

if save_filename is not None:

	with open(save_filename, "a+") as f:
		tmp = np.array2string(labels, precision=2, separator=',')
		f.write(tmp+"\n")
		f.closed
	
	if labels is not None:
		with open(save_truth_filename, "a+") as f:
			tmp = np.array2string(env.trueLabel(), precision=2, separator=',')
			f.write(tmp+"\n")
			f.closed




