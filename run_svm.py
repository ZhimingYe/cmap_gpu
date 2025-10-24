# %%
import pandas as pd

# %%
train_data = pd.read_csv('/root/k3_step1_train.csv', index_col=0)  # should have label column
test_data = pd.read_csv('/root/k3_step1_test.csv', index_col=0)
# %%
params = tune_svm_parameters(
    train_data, test_data,
    scale=True,
    class_weight=True,
    kernel="rbf",
    verbose=True,
    cross_para=[4],
)

# %%

params

# %%
# 预测
best_cost = params['cross_4']['cost']
best_gamma = params['cross_4']['gamma']
predictions_1F = PredictDomain(
    train_set=train_data,
    test_set=test_data,
    scale=True,
    class_weight=True,
    cost=best_cost,
    gamma=best_gamma,
    kernel='rbf',
    st_svm=False,
    verbose=True
)
# %%
predictions_1T = PredictDomain(
    train_set=train_data,
    test_set=test_data,
    scale=True,
    class_weight=True,
    cost=best_cost,
    gamma=best_gamma,
    kernel='rbf',
    st_svm=True,
    verbose=True
)
# %%
# import dill
# with open('predictions_k4.pkl', 'wb') as f:
#     dill.dump({'predictions_1F': predictions_1F, 
#                'predictions_1T': predictions_1T}, f)
# %%
