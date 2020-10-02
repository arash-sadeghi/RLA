from header import SUPERVISOR
if __name__ == "__main__":
    sup=SUPERVISOR()
    sup.generateRobots()
    # sup.swarm[0].PRINTER()
    # sup.PRINTER()
    while True:
        sup.visualize()
        sup.checkCollision()
        sup.moveAll()
        # exit()