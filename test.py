
def CalcOutput(input_size, filter_size, stride, padding):
    return (input_size - filter_size +  padding * 2 // stride) + 1

res = CalcOutput(32, 5, 2, 0)
print(res)