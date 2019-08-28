'''
TODO:
    1.  增加更多的复杂形状障碍物，以及特殊场景（房间等）
    2.  小车的形状以及相应mask
'''
import pygame
import sys

def load():
    OBS_PATH = (
        'assets/obs/obs_40.png',
        'assets/obs/obs_50.png',
        'assets/obs/obs_60.png'
    )
    CAR_PATH = (
        'assets/car/car_30_circle.png',
        'assets/car/car_40_circle.png'
    )
    MASK_PATH = (
        'assets/car/mask_30_circle.png',
        'assets/car/mask_40_circle.png'
    )
    GOAL_PATH = (
        'assets/goal/goal_30_200_0_0.png',
        'assets/goal/goal_30_0_200_0.png',
        'assets/goal/goal_30_0_0_200.png',
        'assets/goal/goal_30_200_200_0.png',
        'assets/goal/goal_30_200_0_200.png',
        'assets/goal/goal_30_0_200_200.png'
    )
    GOAL_MASK_PATH = (
        'assets/goal/goal_mask_30.png',
    )

    IMAGES = {}

    IMAGES['car'] = (
        pygame.image.load(CAR_PATH[0]).convert_alpha(),
        pygame.image.load(CAR_PATH[1]).convert_alpha()
    )

    IMAGES['obs'] = (
        pygame.image.load(OBS_PATH[0]).convert_alpha(),
        pygame.image.load(OBS_PATH[1]).convert_alpha(),
        pygame.image.load(OBS_PATH[2]).convert_alpha()
    )

    IMAGES['mask'] = (
        pygame.image.load(MASK_PATH[0]).convert_alpha(),
        pygame.image.load(MASK_PATH[1]).convert_alpha()
    )
    IMAGES['goal'] = (
        pygame.image.load(GOAL_PATH[0]).convert_alpha(),
        pygame.image.load(GOAL_PATH[1]).convert_alpha(),
        pygame.image.load(GOAL_PATH[2]).convert_alpha(),
        pygame.image.load(GOAL_PATH[3]).convert_alpha(),
        pygame.image.load(GOAL_PATH[4]).convert_alpha(),
        pygame.image.load(GOAL_PATH[5]).convert_alpha()
    )
    IMAGES['goal_mask'] = (
        pygame.image.load(GOAL_MASK_PATH[0]).convert_alpha()
    )
    return IMAGES

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((100, 100))
    IMAGES = load()
    print(IMAGES['car'][0].get_size())
    