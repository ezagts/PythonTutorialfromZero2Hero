course = "Python Programming"
print(len(course))
print(course.count('o', 1))
print(course.find('o'))
print((course[int(len(course) / 2.8):-2]))
print("Python \\Programming\\")
first_name = "Ezana"
last_name = "Girma"

print(f"{first_name} Age = {3 + 2}")
print(course.find("eza"))
print("hi↻")
print(bool(0))

# Conditional statement if statement
course_name = input("Enter the course name: \t")
if len(course_name) > 10:
    print(course)
    print(len(course_name))
# if we have multiple conditions, we can use an if - else statement in python 'elif'
# we can add down below
elif len(course_name) < 10:
    print("Is not the course name try again")
else:
    print("Sorry")
message = "Good" if len(course_name) > 10 else "Bad"
print(message)
print("Done")
