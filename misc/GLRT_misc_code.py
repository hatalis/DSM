# print(classification_report(y_true, y_pred))
# print(accuracy_score(y_true, y_pred))


# mu = residual_test.mean()
# test_statistic = mu
# threshold = np.sqrt(var/N_test)*norm.isf(P_FA)
# if test_statistic > threshold:
#     y_test_prediction = 1
# else:
#     y_test_prediction = 0

# print('threshold = ',threshold)
# print('test_statistic = ', mu)
# print(y_test_prediction)
#
# for t in range(1,N_test+1):
#     threshold = np.sqrt(var / N_test) * norm.isf(P_FA)
#     test_statistic = np.mean(residual_test[:t])
#     if test_statistic > threshold:
#         print(t)

'''
    Q-function = norm.sf(x)
    inverse Q-function = norm.isf(x)
'''
# P_FA = np.arange(0,1,0.01)
# plt.figure()
# for A in range(10):
#     P_D = norm.sf(norm.isf(P_FA) - np.sqrt((N_test * A ** 2) / var))
#     plt.plot(P_FA,P_D, label= 'A = ' + str(A))
# plt.legend(framealpha = 1)
# plt.xlabel('$P_{FA}$')
# plt.ylabel('$P_{D}$')
# plt.title('ROC Curve')
# plt.grid(linestyle='--', linewidth='0.5')