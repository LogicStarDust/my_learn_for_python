def printImgFromMNIST(features, lable,index):
    for i in features[index]:
        for j in i:
            if j == 0:
                print("---", end="")
            else:
                print("xxx", end="")
        print()
    print("label=", lable[index])
