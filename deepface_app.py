from deepface import DeepFace



image1_path="data/obama_01.jpg"
image2_path="data/obama_02.jpg"

model = DeepFace.build_model("Facenet")
result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name="Facenet")
print(result)

