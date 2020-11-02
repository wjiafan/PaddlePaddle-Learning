import paddle.fluid as fluid

# Tensor操作

# 使用Fluid创建5个元素的一维数组,其中每个元素都为1

# 定义数组维度及数据类型，可以修改shape参数定义在任意大小的数组
data = fluid.layers.ones(shape=[5],dtype='int64')
# 在CPU上执行运算
place = fluid.CPUPlace()
# 创建执行器
exe = fluid.Executor(place)
# 执行计算
ones_result = exe.run(
    fluid.default_main_program(),
    fetch_list=[data],
    return_numpy=True
)

print(ones_result[0])


add = fluid.layers.elementwise_add(data,data)
# 定义运算场所
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 执行计算
add_result = exe.run(
    fluid.default_main_program(),
    fetch_list=[add],
    return_numpy=True
)

print(add_result[0])
