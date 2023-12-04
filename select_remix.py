import pandas as pd

## The csv result of Remix
remix_path = 'weights/results_C16_dataset_c16_low99_v0_remix_1class.csv'
# remix_path = 'weights/results_Lung_remix_2class.csv'
df = pd.read_csv(remix_path)

## select model
# df = df[df['method'].str.contains('dsmil')]
df = df[df['method'].str.contains('frmil')]

if 'Lung' in remix_path:
    val_df = df.loc[:,['val_acc', 'val_auc[0]', 'val_auc[1]', 'val_prauc[0]', 'val_prauc[1]']]
    test_df = df.loc[:,['acc', 'auc[0]', 'auc[1]', 'prauc[0]', 'prauc[1]']]
else:
    val_df = df.loc[:,['val_acc', 'val_auc', 'val_prauc']]
    test_df = df.loc[:,['acc', 'auc', 'prauc']]

val_df['val_score'] = val_df.mean(axis=1)
print(val_df)
max_index = val_df['val_score'].argmax()

if 'Lung' in remix_path:
    test_df['avg_auc'] = test_df.loc[:, ['auc[0]', 'auc[1]']].mean(axis=1)
    test_df['avg_prauc'] = test_df.loc[:, ['prauc[0]', 'prauc[1]']].mean(axis=1)

print(val_df.sort_values(by = 'val_score', ascending=False).index)
print(test_df.loc[val_df.sort_values(by = 'val_score', ascending=False).index, :])
print(test_df.iloc[max_index, :])
