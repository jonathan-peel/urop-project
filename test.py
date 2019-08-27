"""Module just for messing around and testing things."""

# Test global variables
# k = 'some string'
# var_1 = 'input me'


# def foo(var_1):
#     print(k + var_1)

# foo(var_1)

"""Test complex control statements."""
# cont = input('Continue? (y or n): ')

# while cont == 'y' or cont == 'n':
#     print('correct entry')
#     cont = input('Continue? (y or n): ')
# This works. I think each expression separated by an 'or' needs to be 
# individually evaluateable.

WM_NUMBER = 2
WM_PATHS = ['str1', 'str2', 'str3']
is_tuple = (WM_PATHS, WM_NUMBER)

for image_path, i in (WM_PATHS, range(WM_NUMBER)):
    print(image_path, i)

