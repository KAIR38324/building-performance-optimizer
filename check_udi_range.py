import pandas as pd

# 检查UDI数据的实际范围
df = pd.read_csv('帕累托解集.csv', encoding='utf-8')
print('UDI统计信息:')
print(df['UDI'].describe())
print(f'\n实际范围: {df["UDI"].min():.2f} - {df["UDI"].max():.2f}')
print(f'标准差: {df["UDI"].std():.2f}')
print(f'\n分位数:')
print(f'25%: {df["UDI"].quantile(0.25):.2f}')
print(f'75%: {df["UDI"].quantile(0.75):.2f}')