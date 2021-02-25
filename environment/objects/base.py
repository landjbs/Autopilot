'''
Base object for populating map.
'''

from config import static_config


class BaseObject:
    '''
    Base object inherited by all non-agent map population.
    Args:
        static:     (Bool) whether moves.
        height:     (int) height of object in pixels
        width:      (int) width of object in pixels
        color:      (tuple of int) RBG color on 255
        behavior:   (Behavior) optional behavior obj defining object's behavior
        rules:      (Rules) Todo
    '''

    def __init__(
            self, static: bool, height: int, width: int, color: tuple,
            behavior: Behavior = None
        ):
        # assertions
        # assert isinstance(static, bool), 'Static expected type bool'
        assert (height > static_config.frame_height), (
            f'height cannot be greater than frame_height, '
            f'{static_config.frame_height}.'
        )
        assert (width > static_config.frame_width), (
            f'width cannot be greater than frame_width, '
            f'{static_config.frame_width}.'
        )
        assert all(color in range(0, 255)), (
            'all values in color must be in range [0, 255]'
        )
        # cache
        self.static = bool(static)
        self.height = int(height)
        self.width = int(width)
        self.color = color
        if not self.static:
            self.behavior = behavior
