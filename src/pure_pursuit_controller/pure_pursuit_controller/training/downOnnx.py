import onnx
model = onnx.load("dagger_model_sim_2.onnx")
model.ir_version = 9  # Ép định dạng file về bản tương thích với Docker
onnx.save(model, "dagger_model_sim_2.onnx")
