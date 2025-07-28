import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import math
import warnings
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")


# 1. 数据加载和预处理
def load_data(file_path, continuous_cols=None, categorical_cols=None):
    """加载Excel数据并分离特征和标签

    参数:
        file_path: Excel文件路径
        continuous_cols: 手动指定的连续型特征列名列表
        categorical_cols: 手动指定的分类型特征列名列表
    """
    df = pd.read_excel(file_path)

    # 第一列是ID，最后四列是标签，中间是特征
    id_col = df.columns[0]
    label_cols = df.columns[-4:]
    feature_cols = df.columns[1:-4]

    # 分离数据
    ids = df[id_col]
    features = df[feature_cols]
    labels = df[label_cols]

    # 确定每列的类型
    if continuous_cols is None and categorical_cols is None:
        # 自动判断类型
        col_types = {}
        for col in features.columns:
            if features[col].dtype in ['int64', 'float64']:
                # 检查是否是分类数据（少于10个唯一值）
                if len(features[col].dropna().unique()) < 10:
                    col_types[col] = 'categorical'
                else:
                    col_types[col] = 'continuous'
            else:
                col_types[col] = 'categorical'
    else:
        # 使用手动指定的类型
        col_types = {}
        for col in features.columns:
            if continuous_cols and col in continuous_cols:
                col_types[col] = 'continuous'
            elif categorical_cols and col in categorical_cols:
                col_types[col] = 'categorical'
            else:
                # 自动判断
                if features[col].dtype in ['int64', 'float64']:
                    if len(features[col].dropna().unique()) < 10:
                        col_types[col] = 'categorical'
                    else:
                        col_types[col] = 'continuous'
                else:
                    col_types[col] = 'categorical'

    return ids, features, labels, col_types


# 2. MissForest实现
class MissForest:
    """MissForest缺失值填充算法实现

    可调参数:
        max_iter: 最大迭代次数 (默认: 10)
        n_estimators: 随机森林的树数量 (默认: 100)
        random_state: 随机种子 (默认: 42)
        initial_guess: 初始填充策略
            - 'mean' 连续变量用均值，分类变量用众数 (默认)
            - 'median' 连续变量用中位数，分类变量用众数
            - 'random' 随机填充
        verbose: 是否显示进度条 (默认: True)
    """

    def __init__(self, max_iter=10, n_estimators=100, random_state=42,
                 initial_guess='mean', verbose=True):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.initial_guess = initial_guess
        self.verbose = verbose

    def fit_transform(self, X, col_types):
        """填充缺失值

        参数:
            X: 包含缺失值的DataFrame
            col_types: 字典，指定每列的类型 ('continuous' 或 'categorical')
        """
        # 复制数据以避免修改原始数据
        X_filled = X.copy()

        # 记录每列的缺失位置
        missing_mask = X_filled.isnull()

        # 初始化填充
        if self.initial_guess == 'mean':
            for col in X_filled.columns:
                if col_types[col] == 'continuous':
                    X_filled[col].fillna(X_filled[col].mean(), inplace=True)
                else:
                    X_filled[col].fillna(X_filled[col].mode()[0], inplace=True)
        elif self.initial_guess == 'median':
            for col in X_filled.columns:
                if col_types[col] == 'continuous':
                    X_filled[col].fillna(X_filled[col].median(), inplace=True)
                else:
                    X_filled[col].fillna(X_filled[col].mode()[0], inplace=True)
        elif self.initial_guess == 'random':
            for col in X_filled.columns:
                if col_types[col] == 'continuous':
                    non_missing = X_filled[col].dropna()
                    if len(non_missing) > 0:
                        X_filled[col].fillna(np.random.choice(non_missing), inplace=True)
                else:
                    non_missing = X_filled[col].dropna()
                    if len(non_missing) > 0:
                        X_filled[col].fillna(np.random.choice(non_missing), inplace=True)

        # 标签编码分类变量
        encoders = {}
        for col in X_filled.columns:
            if col_types[col] == 'categorical':
                le = LabelEncoder()
                X_filled[col] = le.fit_transform(X_filled[col].astype(str))
                encoders[col] = le

        # 迭代填充
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="MissForest Iterations")

        for _ in iterator:
            X_filled_old = X_filled.copy()

            # 对每列进行填充
            for col in X_filled.columns:
                if missing_mask[col].sum() == 0:
                    continue  # 没有缺失值，跳过

                # 获取当前列和其他列
                other_cols = [c for c in X_filled.columns if c != col]
                X_other = X_filled[other_cols]
                y = X_filled[col]

                # 分割为有缺失和无缺失的数据
                missing_idx = missing_mask[col]
                train_idx = ~missing_idx
                test_idx = missing_idx

                X_train = X_other[train_idx]
                y_train = y[train_idx]
                X_test = X_other[test_idx]

                if col_types[col] == 'continuous':
                    # 回归问题
                    model = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                else:
                    # 分类问题
                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                # 填充预测值
                X_filled.loc[test_idx, col] = preds

            # 检查收敛
            diff = ((X_filled - X_filled_old) ** 2).sum().sum()
            if diff < 1e-6:
                if self.verbose:
                    print(f"\n提前收敛于迭代 {_ + 1}")
                break

        # 反向转换分类变量
        for col in encoders:
            le = encoders[col]
            X_filled[col] = le.inverse_transform(X_filled[col].astype(int))
            # 转换回原始数据类型
            if X[col].dtype == 'object':
                X_filled[col] = X_filled[col].astype(str)
            else:
                X_filled[col] = X_filled[col].astype(X[col].dtype)

        return X_filled


# 3. 评估函数
def evaluate_imputation(original, imputed, mask, col_types):
    """评估填充效果

    参数:
        original: 原始完整数据
        imputed: 填充后的数据
        mask: 缺失值掩码 (True表示该位置是缺失值)
        col_types: 列类型字典

    返回:
        包含评估指标的字典 (NRMSE, MAE, PFC)
    """
    # 只评估原来缺失的值
    original_missing = original[mask]
    imputed_missing = imputed[mask]

    # 分离连续和分类变量
    continuous_cols = [col for col in original.columns if col_types.get(col) == 'continuous']
    categorical_cols = [col for col in original.columns if col_types.get(col) == 'categorical']

    # 初始化结果
    results = {
        'NRMSE': None,
        'MAE': None,
        'PFC': None
    }

    # 连续变量评估 (NRMSE和MAE)
    if len(continuous_cols) > 0:
        original_cont = original_missing[continuous_cols]
        imputed_cont = imputed_missing[continuous_cols]

        # 计算RMSE
        mse = mean_squared_error(original_cont, imputed_cont)
        rmse = math.sqrt(mse)

        # 计算NRMSE (除以原始值的标准差)
        std_dev = original[continuous_cols].std().mean()
        nrmse = rmse / std_dev if std_dev != 0 else float('inf')

        # 计算MAE
        mae = mean_absolute_error(original_cont, imputed_cont)

        results['NRMSE'] = nrmse
        results['MAE'] = mae

    # 分类变量评估 (PFC)
    if len(categorical_cols) > 0:
        original_cat = original_missing[categorical_cols]
        imputed_cat = imputed_missing[categorical_cols]

        # 计算错分类比例
        pfc = (original_cat != imputed_cat).sum().sum() / (original_cat.size)

        results['PFC'] = pfc

    return results


# 4. 主函数
def main(file_path, test_ratio=0.1, continuous_cols=None, categorical_cols=None,
         max_iter=10, n_estimators=100, random_state=42, initial_guess='mean'):
    """主流程

    可调参数:
        file_path: Excel文件路径
        test_ratio: 用于评估的数据比例 (默认: 0.1)
        continuous_cols: 手动指定的连续型特征列名列表
        categorical_cols: 手动指定的分类型特征列名列表
        max_iter: MissForest最大迭代次数 (默认: 10)
        n_estimators: 随机森林的树数量 (默认: 100)
        random_state: 随机种子 (默认: 42)
        initial_guess: 初始填充策略 ('mean', 'median' 或 'random') (默认: 'mean')
    """
    # 1. 加载数据并获取列类型
    ids, features, labels, col_types = load_data(
        file_path,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols
    )

    print("\n列类型信息:")
    for col, typ in col_types.items():
        print(f"{col}: {typ}")

    # 2. 创建缺失值掩码 (用于评估)
    missing_mask = features.isnull()

    if missing_mask.sum().sum() == 0:
        print("\n数据中没有缺失值，将人为添加缺失值用于评估")
        # 人为添加缺失值用于评估
        np.random.seed(random_state)
        test_mask = np.random.random(features.shape) < test_ratio
        features_with_nan = features.copy()
        features_with_nan[test_mask] = np.nan
        original_features = features.copy()
    else:
        print("\n数据中存在真实缺失值")
        # 为了评估，我们保留一部分已知值
        np.random.seed(random_state)
        test_mask = features.notnull() & (np.random.random(features.shape) < test_ratio)
        features_with_nan = features.copy()
        features_with_nan[test_mask] = np.nan
        original_features = features.copy()

    # 3. 使用MissForest填充缺失值
    print("\n开始使用MissForest填充缺失值...")
    mf = MissForest(
        max_iter=max_iter,
        n_estimators=n_estimators,
        random_state=random_state,
        initial_guess=initial_guess,
        verbose=True
    )
    features_filled = mf.fit_transform(features_with_nan, col_types)
    print("缺失值填充完成!")

    # 4. 评估填充效果
    print("\n评估填充效果...")
    eval_results = evaluate_imputation(
        original_features,
        features_filled,
        test_mask,
        col_types
    )

    print("\n评估结果:")
    if eval_results['NRMSE'] is not None:
        print(f"标准化均方根误差 (NRMSE): {eval_results['NRMSE']:.4f}")
    if eval_results['MAE'] is not None:
        print(f"平均绝对误差 (MAE): {eval_results['MAE']:.4f}")
    if eval_results['PFC'] is not None:
        print(f"错分类比例 (PFC): {eval_results['PFC']:.4f}")

    # 5. 保存填充后的数据
    output_df = pd.concat([ids, features_filled, labels], axis=1)
    output_path = file_path.replace('.xlsx', '_filled.xlsx')
    output_df.to_excel(output_path, index=False)
    print(f"\n填充后的数据已保存到: {output_path}")


if __name__ == "__main__":
    # 替换为你的Excel文件路径
    excel_file = "D:\\研二2\\论文撰写\\3.30正式数据\\箱线图处理异常值.xlsx"  # 请修改为你的文件路径

    # ==============================================
    # 可调整的参数 (根据需要修改这些值)
    # ==============================================

    # 手动指定列类型 (如果不指定，将自动判断)
    continuous_cols = ["嗜碱性粒细胞数量（1000 个细胞/μL）", "成功测量次数", "嗜酸性粒细胞数量（1000 个细胞/μL）",
                         "有核红细胞", "工作日的正常唤醒时间（时）"]  # 例如: ['age', 'income']
    categorical_cols = ["过去 12 个月喝啤酒的频率（0去年从来没有，1每天，2几乎每天，3每周 3 至 4 次，4每周 2 次，5每周一次，6每月 2 至 3 次，7每月一次，8去年 7 到 11 次，9去年 3 至 6 次，10去年 1 至 2 次，拒绝，不知道）",
        "每周、每月或每年在一天内喝了多少天（0：去年从来没有，1：每天，2：几乎每天，3：每周 3 至 4 次，4：每周 2 次，5：每周一次，6：每月 2 至 3 次，7：每月一次，8：去年 7 到 11 次，9：去年 3 至 6 次，10：去年 1 至 2 次，拒绝，不知道）"]  # 强制视为分类型的特征
  # 例如: ['gender', 'education']


    # MissForest参数
    max_iter = 10  # 最大迭代次数
    n_estimators = 100  # 随机森林的树数量
    random_state = 42  # 随机种子
    initial_guess = 'mean'  # 初始填充策略 ('mean', 'median' 或 'random')

    # 评估参数
    test_ratio = 0.1  # 用于评估的数据比例

    # ==============================================

    main(
        excel_file,
        test_ratio=test_ratio,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        max_iter=max_iter,
        n_estimators=n_estimators,
        random_state=random_state,
        initial_guess=initial_guess
    )