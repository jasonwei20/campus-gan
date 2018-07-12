import matplotlib.pyplot as plt
x = [4, 3, 2, 1]
y = [10/128, 12/128, 12/128, 45/128]
plt.plot(x, y)
plt.title('Misclassifications Rates of SVM on Resnet Block Embeddings')
plt.xlabel('Resnet Block Number')
plt.ylabel('Training Error')
plt.savefig('svm_misclassification', dpi=400)