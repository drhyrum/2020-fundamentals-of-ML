print("Think of one of these: giraffe, seal, chicken, penguin, elephant")
if input("is it aquatic?").lower() == "y":

    if input("does it lay eggs?").lower() == "y":
        print("penguin")

    else:
        print("seal")

elif input("does it lay eggs?").lower() == "y":
    print("chicken")

elif input("does it have 4 legs?").lower() == "y":

    if input("does it have a trunk?").lower() == "y":
        print("elephant")

    else:
        print("giraffe")
