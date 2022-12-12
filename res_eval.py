import matplotlib.pyplot as plt
def loadx(path):
    list = []
    with open('results/'+ path,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            list.append(int(line[line.index('epoch')+6:line.index(', loss')]))
    return list

def load(path):
    list = []
    with open('results/'+ path,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            list.append(float(line[line.index('test acc:')+10:line.index('test acc:')+15]))
    return list

def eval_dice():
    x = loadx('DICEattack0.txt')
    res_dice_0 = load('DICEattack0.txt')
    res_dice_100 = load('DICEattack100.txt')
    res_dice_500 = load('DICEattack500.txt')
    plt.plot(x,res_dice_0,'g',label="res_dice_0")
    plt.plot(x,res_dice_100,'c',label="res_dice_100")
    plt.plot(x,res_dice_500,'b',label="res_dice_500")
    plt.xlabel("epoh")
    plt.ylabel("accuracy")
    plt.legend(loc = "best")
    plt.savefig('results/eval_dice.jpg')
    plt.show()

def eval_randomadd():
    x = loadx('Randomattack0add.txt')
    res_random_0 = load('Randomattack0add.txt')
    res_random_100 = load('Randomattack100add.txt')
    res_random_500 = load('Randomattack500add.txt')
    res_random_1000 = load('Randomattack1000add.txt')
    plt.plot(x,res_random_0,'g',label="res_random_0_add")
    plt.plot(x,res_random_100,'c',label="res_random_100_add")
    plt.plot(x,res_random_500,'b',label="res_random_500_add")
    plt.plot(x,res_random_1000,'r',label="res_random_1000_add")
    plt.xlabel("epoh")
    plt.ylabel("accuracy")
    plt.legend(loc = "best")
    plt.savefig('results/eval_randomadd.jpg')
    plt.show()

def eval_randomremove():
    x = loadx('Randomattack0remove.txt')
    res_random_0 = load('Randomattack0remove.txt')
    res_random_100 = load('Randomattack100remove.txt')
    res_random_500 = load('Randomattack500remove.txt')
    res_random_1000 = load('Randomattack1000remove.txt')
    plt.plot(x,res_random_0,'g',label="res_random_0_remove")
    plt.plot(x,res_random_100,'c',label="res_random_100_remove")
    plt.plot(x,res_random_500,'b',label="res_random_500_remove")
    plt.plot(x,res_random_1000,'r',label="res_random_1000_remove")
    plt.xlabel("epoh")
    plt.ylabel("accuracy")
    plt.legend(loc = "best")
    plt.savefig('results/eval_randomremove.jpg')
    plt.show()

def eval_NodeEmbeddingremove():
    x = loadx('NodeEmbedding100remove.txt')
    res_ne_1 = load('NodeEmbedding1remove.txt')
    res_ne_100 = load('NodeEmbedding100remove.txt')
    res_ne_500 = load('NodeEmbedding500remove.txt')
    res_ne_1000 = load('NodeEmbedding1000remove.txt')
    res_ne_2000 = load('NodeEmbedding2000remove.txt')
    plt.plot(x,res_ne_1,'g',label="res_ne_1_remove")
    plt.plot(x,res_ne_100,'c',label="res_ne_100_remove")
    plt.plot(x,res_ne_500,'b',label="res_ne_500_remove")
    plt.plot(x,res_ne_1000,'r',label="res_ne_1000_remove")
    plt.plot(x,res_ne_2000,'m',label="res_ne_2000_remove")
    plt.xlabel("epoh")
    plt.ylabel("accuracy")
    plt.legend(loc = "best")
    plt.savefig('results/eval_NodeEmbedding_remove.jpg')
    plt.show()

def eval_NodeEmbeddingadd():
    x = loadx('NodeEmbedding100add.txt')
    res_ne_1 = load('NodeEmbedding1add.txt')
    res_ne_100 = load('NodeEmbedding100add.txt')
    res_ne_500 = load('NodeEmbedding500add.txt')
    res_ne_1000 = load('NodeEmbedding1000add.txt')
    res_ne_2000 = load('NodeEmbedding2000add.txt')
    plt.plot(x,res_ne_1,'g',label="res_ne_1_add")
    plt.plot(x,res_ne_100,'c',label="res_ne_100_add")
    plt.plot(x,res_ne_500,'b',label="res_ne_500_add")
    plt.plot(x,res_ne_1000,'r',label="res_ne_1000_add")
    plt.plot(x,res_ne_2000,'m',label="res_ne_2000_add")
    plt.xlabel("epoh")
    plt.ylabel("accuracy")
    plt.legend(loc = "best")
    plt.savefig('results/eval_NodeEmbedding_add.jpg')
    plt.show()


eval_dice()
eval_randomadd()
eval_randomremove()
eval_NodeEmbeddingremove()
eval_NodeEmbeddingadd()
