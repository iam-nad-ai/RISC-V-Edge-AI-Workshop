clf.flit(X_train_scaled, y_train)

y_prod = clf.predict(X_test_scaled)
print(“Accuracy”, accuracy_score(y_test, y_prod))


import seabird as ins
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=‘d’, cmap=‘Blues’, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel(‘Predicted label’)
plt.ylabel(’True Label’)
plt.title(‘Confusion Matrix’)
plt.show()

with open("svm_model.h", "w") as f:
  f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
  f.write(f"#define NUM_FEATURES {weights.shape[1]}\n\n")

  f.write("double weights[NUM_CLASSES][NUM_FEATURES] {\n")
  for row in weights:
    f.write("  {" + ", ".join(f"{v:.10f}" for v in row) + "}, \n")
  f.write("}; \n\n")

  f.write("double bias[NUM_CLASSES] = {" + ", ".join(f"{b:.10f}" for b in biases) + "};\n")

print("Exported SVM model to svm_model.h")


Mean = scaler.mean_
Scale =scaler.scale_
F.write(f”#define NUM_FEATURES {len(mean)}\n\n”)
F.write(“double mean[NUM_FEATURES] = {\n”)
F.write(“ “ + “, “.join(f”{m:.10f}” for m in mean) + “\n};\n\n”)

F.write(“double scale[NUM_FEATURES] = {\n”)
f.write(“ “ + “, “.join(f”{s:.10f}” for s in scale) + “\n};\n”)

print(“exported scalar parameters to scalar.h”)
