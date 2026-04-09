import pygame
from time import sleep

pygame.init()
pygame.joystick.init()

js = pygame.joystick.Joystick(0)
js.init()

while True:
    pygame.event.pump()

    axes = [js.get_axis(i) for i in range(js.get_numaxes())]
    buttons = [js.get_button(i) for i in range(js.get_numbuttons())]

    

    print("Axes:", axes)
    print("Buttons:", buttons)
    sleep(0.1)