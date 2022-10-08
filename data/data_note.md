- ### 对于一个类只有一条序列的数据（在sklearn方法的train_test_split方法中要求每一类至少有两条数据才可分别放入训练集和测试集）

**A**：这种数据全部放在训练集 引入更多先验知识 
同时也提供了剔除每一类只有一条数据 并且使分类标签保留后四位（原始protein_clean.csv文件标签太长 按照数字大小来分类会产生错误） 产生protein.csv文件
具体步骤如下：
- 在protein_clean.csv文件中利用=COUNTIF(F:F,F2)【F为整列 F2为第一个数据 然后表中往下拉即可】 即可得出每一个类别的计数 删除类别数为1的即可
- 之后使用RIGHT函数 另开新列 保证label的数字不能太大（取后四位即可 即可生成protein.csv文件用于测试分类效果）

- ### 当直接采用 family/superfamily/fold 的类别数作为中的output_size或num_class时 会出现造成device-side assert报错
**A**：原因是 三个模型都是用了pytorch中的DataLoader方法 当设定类别为 *N* 时 映射标签序号为 *0~N*    
但是 family/superfamily/fold 的标签序号并非简单设定为 *0~N* （而class仅有5类，为此种映射方式）  
**具体可见data/protein_xx.csv文件**


Note:   
对family\superfamily\fold的单任务分类效果比较差 loss无法稳定降低到2以下 准确率较低（原因是 相当多类别只有极少的训练序列 具体可见DeepSF文章2.1节）
