def a():
    c = 1
    def x(variable):
        return variable + c
    return x

fun = a()
print(fun(3))
