from bee_foraging_model.run import *

def main():
    model = BeeForagingModel(3333, 10, flower_open= 7 * 3600, flower_closed= 9 * 3600, sucrose_concentration=0.25)
    model.run(500000)

if __name__ == '__main__':
    main()