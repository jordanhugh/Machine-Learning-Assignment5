import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyRegressor

def gaussian_kernel(distances): 
    weights = np.exp(-parameter_knn*(distances**2))
    return weights/np.sum(weights)

data = np.array([[-1,0],[0,1],[1,0]])
X = data[:, 0]
X = X.reshape(-1, 1)
y = data[:, 1]
y = y.reshape(-1, 1)
xx = np.linspace(-3.0, 3.0, num=1000)
xx = xx.reshape(-1,1)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Visuation of Data', fontsize=14)
ax.set_xlabel('X1', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.scatter(X, y, marker='o')
plt.savefig('small_dataset_visualisation')
plt.show()

parameters = [0, 1, 5, 10, 25]
fig = plt.figure(figsize=(14, 21))
fig.suptitle('kNN Classifier Predictions', fontsize=14, y=0.9125)
for itr, parameter_knn in enumerate(parameters, 1):
    knn = KNeighborsRegressor(n_neighbors=3, weights=gaussian_kernel)
    knn.fit(X, y.ravel())
    y_knn = knn.predict(xx)
    ax = fig.add_subplot(3, 2, itr)
    ax.set_title('γ = ' + str(parameter_knn), fontsize=14)
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.scatter(X, y, marker='o')
    ax.plot(xx, y_knn)
    ax.legend(['Pred', 'True'], loc='upper right')
plt.savefig('knn_dataset1_paramaters')
plt.show()

models = []
parameters = [0, 1, 5, 10, 25]
fig = plt.figure(figsize=(14, 21))
fig.suptitle('KRR Classifier Predictions', fontsize=14, y=0.9125)
for itr, parameter_krr in enumerate(parameters, 1):
    krr = KernelRidge(alpha=1.0/(2 * 1.0), kernel='rbf', gamma=parameter_krr)
    krr.fit(X, y.ravel())
    models.append(krr)
    y_krr = krr.predict(xx)
    ax = fig.add_subplot(3, 2, itr)
    ax.set_title('γ = ' + str(parameter_krr), fontsize=14)
    ax.set_xlabel('X1 (Normalised)', fontsize=12)
    ax.set_ylabel('y (Normalised)', fontsize=12)
    ax.scatter(X, y, marker='o')
    ax.plot(xx, y_krr)
    ax.legend(['Pred', 'True'], loc='upper right')
plt.savefig('krr_dataset1_paramaters')
plt.show()

print('KRR Parameter Table:')
print('%-13s %-12s' % ('Parameter(γ)', 'Parameter(θ)'))
for itr, model in enumerate(models):
    print('%-13s %-12a' % (parameters[itr], model.dual_coef_))

parameter_krr = 5;

models = []
penalties = [0.1, 1, 1000]
fig = plt.figure(figsize=(14, 14))
fig.suptitle('KRR Classifier Predictions', fontsize=14, y=0.9125)
for itr, penalty in enumerate(penalties, 1):
    krr = KernelRidge(alpha=1.0/(2*penalty), kernel='rbf', gamma=parameter_krr)
    krr.fit(X, y.ravel())
    models.append(krr)
    y_krr = krr.predict(xx)
    ax = fig.add_subplot(2, 2, itr)
    ax.set_title('C = ' + str(penalty), fontsize=14)
    ax.set_xlabel('X1 (Normalised)', fontsize=12)
    ax.set_ylabel('y (Normalised)', fontsize=12)
    ax.scatter(X, y, marker='o')
    ax.plot(xx, y_krr)
    ax.legend(['Pred', 'True'], loc='upper right')
plt.savefig('krr_dataset1_penalties')
plt.show()

print('KRR Penalty Parameter Table:')
print('%-11s %-12s' % ('Penalty(C)', 'Parameter(θ)'))
for itr, model in enumerate(models):
    print('%-11s %-12a' % (penalties[itr], model.dual_coef_))



df = pd.read_csv('week5.csv',comment='#') 
X = np.array(df.iloc[:, 0])
X = X.reshape(-1, 1)
X = scale(X)
y = np.array(df.iloc[:, 1])
y = y.reshape(-1, 1)
y = scale(y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Visuation of Data', fontsize=14)
ax.set_xlabel('X1 (Normalised)', fontsize=12)
ax.set_ylabel('y (Normalised)', fontsize=12)
ax.scatter(X, y, c=y, marker='o', cmap='coolwarm', alpha=0.4)
plt.savefig('large_dataset_visualisation')
plt.show()

models = []
k = len(y)
parameters = [0, 1, 5, 10, 25, 50]
fig = plt.figure(figsize=(14, 21))
fig.suptitle('kNN Classifier Predictions', fontsize=14, y=0.9125)
for itr, parameter_knn in enumerate(parameters, 1):
    knn = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel)
    knn.fit(X, y.ravel())
    models.append(knn)
    y_knn = knn.predict(xx)
    ax = fig.add_subplot(3, 2, itr)
    ax.set_title('γ = ' + str(parameter_knn), fontsize=14)
    ax.set_xlabel('X1 (Normalised)', fontsize=12)
    ax.set_ylabel('y (Normalised)', fontsize=12)
    ax.scatter(X, y, c=y, marker='o', cmap='coolwarm', alpha=0.4)
    ax.plot(xx, y_knn)
    ax.legend(['Pred', 'True'], loc='lower right')
plt.savefig('knn_dataset2_paramaters')
plt.show()

kf = KFold(n_splits=5)
mse = []; mse_std = []
parameters = [0, 1, 5, 10, 25, 50]
for itr, parameter_knn in enumerate(parameters):
    error = []
    knn = KNeighborsRegressor(weights=gaussian_kernel)
    for train, test in kf.split(X):
        knn.set_params(n_neighbors=len(y[test]))
        knn.fit(X[train], y[train].ravel())
        y_knn = knn.predict(X[test])
        error.append(mean_squared_error(y[test], y_knn))
    mse.append(np.array(error).mean())
    mse_std.append(np.array(error).std())

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Optimal Value of Parameter γ', fontsize=14)
ax.set_xlabel('γ', fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
ax.errorbar(parameters, mse, yerr=mse_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(parameters, mse, marker='o')
plt.savefig('knn_parameter_cross_validation')
plt.show()

parameter_knn = 10
print('Optimal Value: %d\n' % (parameter_knn))

knn = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel)
knn.fit(X, y.ravel())
y_knn = knn.predict(xx)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('kNN Classifier', fontsize=14)
ax.set_xlabel('X1 (Normalised)', fontsize=12)
ax.set_ylabel('y (Normalised)', fontsize=12)
ax.scatter(X, y, c=y, marker='o', cmap='coolwarm', alpha=0.4)
ax.plot(xx, y_knn)
ax.legend(['Pred', 'True'], loc='lower right')
plt.savefig('knn_classifier')
plt.show()

models = []
parameters = [0, 1, 5, 10, 25]
fig = plt.figure(figsize=(14, 21))
fig.suptitle('KRR Classifier Predictions', fontsize=14, y=0.9125)
for itr, parameter_krr in enumerate(parameters, 1):
    krr = KernelRidge(alpha=1.0/(2 * 1.0), kernel='rbf', gamma=parameter_krr)
    krr.fit(X, y.ravel())
    models.append(krr)
    y_krr = krr.predict(xx)
    ax = fig.add_subplot(3, 2, itr)
    ax.set_title('γ = ' + str(parameter_krr), fontsize=14)
    ax.set_xlabel('X1 (Normalised)', fontsize=12)
    ax.set_ylabel('y (Normalised)', fontsize=12)
    ax.scatter(X, y, c=y, marker='o', cmap='coolwarm', alpha=0.4)
    ax.plot(xx, y_krr)
    ax.legend(['Pred', 'True'], loc='lower right')
plt.savefig('krr_dataset2_paramaters')
plt.show()

kf = KFold(n_splits=5)
mse = []; mse_std = []
for model in models:
    error = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train].ravel())
        y_pred = model.predict(X[test])
        error.append(mean_squared_error(y[test], y_pred))
    mse.append(np.array(error).mean())
    mse_std.append(np.array(error).std())

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Optimal Value of Parameter γ', fontsize=14)
ax.set_xlabel('γ', fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
ax.errorbar(parameters, mse, yerr=mse_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(parameters, mse, marker='o')
plt.savefig('krr_parameter_cross_validation')
plt.show()

parameter_krr = 1
print('Optimal Value: %d\n' % (parameter_krr))

models = []
penalties = [0.1, 1, 1000]
fig = plt.figure(figsize=(14, 14))
fig.suptitle('KRR Classifier Predictions', fontsize=14, y=0.9125)
for itr, penalty in enumerate(penalties, 1):
    krr = KernelRidge(alpha=1.0/(2*penalty), kernel='rbf', gamma=parameter_krr)
    krr.fit(X, y.ravel())
    models.append(krr)
    y_krr = krr.predict(xx)
    ax = fig.add_subplot(2, 2, itr)
    ax.set_title('C = ' + str(penalty), fontsize=14)
    ax.set_xlabel('X1 (Normalised)', fontsize=12)
    ax.set_ylabel('y (Normalised)', fontsize=12)
    ax.scatter(X, y, c=y, marker='o', cmap='coolwarm', alpha=0.4)
    ax.plot(xx, y_krr)
    ax.legend(['Pred', 'True'], loc='lower right')
plt.savefig('krr_dataset2_penalties')
plt.show()

models = []
penalties = [0.1, 0.5, 1, 5, 10, 50]
for penalty in penalties:
    krr = KernelRidge(alpha=1.0/(2*penalty), kernel='rbf', gamma=parameter_krr)
    krr.fit(X, y.ravel())
    models.append(krr)

kf = KFold(n_splits=5)
mse = []; mse_std = []
for model in models:
    error = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train].ravel())
        y_pred = model.predict(X[test])
        error.append(mean_squared_error(y[test], y_pred))
    mse.append(np.array(error).mean())
    mse_std.append(np.array(error).std())

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Optimal Value of Parameter C', fontsize=14)
ax.set_xlabel('C', fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
ax.errorbar(penalties, mse, yerr=mse_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(penalties, mse, marker='o')
plt.savefig('krr_penalty_cross_validation')
plt.show()

penalty_krr = 5
print('Optimal Value: %d\n' % (penalty_krr))

krr = KernelRidge(alpha=1.0/(2*penalty_krr), kernel='rbf', gamma=parameter_krr)
krr.fit(X, y.ravel())
y_krr = krr.predict(xx)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('KRR Classifier', fontsize=14)
ax.set_xlabel('X1 (Normalised)', fontsize=12)
ax.set_ylabel('y (Normalised)', fontsize=12)
ax.scatter(X, y, c=y, marker='o', cmap='coolwarm', alpha=0.4)
ax.plot(xx, y_krr)
ax.legend(['Pred', 'True'], loc='lower right')
plt.savefig('krr_classifier')
plt.show()

knn = KNeighborsRegressor(n_neighbors=len(y[test]), weights=gaussian_kernel)
krr = KernelRidge(alpha=1.0/(2*penalty_krr), kernel='rbf', gamma=parameter_krr)
dummy = DummyRegressor(strategy='mean')

kf = KFold(n_splits=5)
mse = []; mse_std = []
models = [knn, krr, dummy]
for itr, model in enumerate(models):
    error = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train].ravel())
        y_pred = model.predict(X[test])
        error.append(mean_squared_error(y[test], y_pred))
    mse.append(np.array(error).mean())
    mse_std.append(np.array(error).std())

print('Evaluation:')
print('%-12s %-10s %-10s' % ('Classifier', 'MSE', 'STD'))
labels = ['kNN', 'KRR', 'Dummy']
for itr in range(len(mse)):
    print('%-12s %-10f %-10f' % (labels[itr], mse[itr], mse_std[itr]))