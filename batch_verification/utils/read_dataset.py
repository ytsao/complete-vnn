from vnnlib import compat
import onnx
import onnxruntime as ort


def read_onnx_model(onnx_file_path: str) -> onnx.ModelProto:
    onnx_model = onnx.load(onnx_file_path)

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = [i.name for i in sess.get_inputs()]
    outputs = [o.name for o in sess.get_outputs()]
    print("Inputs: ", inputs)
    print("Outputs: ", outputs)

    graph = onnx_model.graph
    all_input = [n for n in graph.input if n.name == inputs[0]][0]
    all_output = [n for n in graph.output if n.name == outputs[0]][0]

    for n in graph.input :
        print(n)
    
    num_inputs = 1
    for d in all_input.type.tensor_type.shape.dim:
        num_inputs *= d.dim_value

    num_outputs = 1
    for d in all_output.type.tensor_type.shape.dim:
        num_outputs *= d.dim_value

    return onnx_model


def read_vnnlib_spec(vnnlib_file_path: str, inputs: int, outputs: int):
    vnnlib_spec = compat.read_vnnlib_simple(vnnlib_file_path, inputs, outputs)

    return vnnlib_spec


def test():
    onnx_model = read_onnx_model("./benchmarks/onnx/ACASXU_run2a_1_1_batch_2000.onnx")
    vnnlib_spec = read_vnnlib_spec("./benchmarks/vnnlib/test_small.vnnlib", 5, 5)

    # print information
    print(vnnlib_spec)


if __name__ == "__main__":
    test()
    print("Done!")