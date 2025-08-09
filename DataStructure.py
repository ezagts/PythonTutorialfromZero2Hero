# Talking about Array in DataStructure

"""
    We have lists of monthly expense
    Jan - 2200
    Feb - 2350
    Mar - 2600
    Apr - 2130
    May - 2190

"""

# Based on the above data, we can answer the following question

# In Feb, How many dollars you spent extra compare to Jan?
months = ["Jan", "Feb", "Mar", "Apr", "May"]
expenses = [2200, 2350, 2600, 2130, 2190]
print(f" I spent extra money on Feb compare to Jan is ${expenses[1] - expenses[0]}")

# To find out the total expense the first quarter

total = 0
for i in range(len(expenses)):
    total += expenses[i]
print(f" The total expense of the first quarter\t${total} ")

# Find out if I spent exactly 2000 dollars in any month

found = ""
for i in range(len(expenses)):
    if expenses[i] == 2000:
        found = months[i]
        print(f"I found out which month i spent 2000 dollar is {found}")
else:
    print(" I didn't spent 2000 dollar any of the month👌")

# I want to add june month I already finished this month expense

months.insert(5, "June")
expenses.insert(5, 1980)
print(f" Each month and their correspond of expense {list(zip(months, expenses))}")

refundApr = 0
for i in range(len(months)):
    if months[i] == "Apr":
        refundApr = expenses[i] - 200
        expenses[i] = refundApr
print(f"The expense of the apps month after refund is {list(zip(months, expenses))}")
