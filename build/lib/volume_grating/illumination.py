from .sources import Source


class Record(object):
    """
    Record class creates an illumination instance that emulates how a hologram will be made. Therefore, it needs two
    light sources as its argument.

    :param source1: an instance of Source subclasses.
    :param source2: an instance of Source subclasses.
    """

    def __init__(self, source1, source2):
        self.source1 = source1
        self.source2 = source2

    @property
    def source1(self):
        return self._source1

    @source1.setter
    def source1(self, value):
        assert isinstance(value, Source) or value is None, 'source1 must be an instance of {} class.'.format(Source)
        self._source1 = value

    @property
    def source2(self):
        return self._source2

    @source2.setter
    def source2(self, value):
        assert isinstance(value, Source) or value is None, ' source2 must be an instance of {} class.'.format(Source)
        self._source2 = value

    def __str__(self):
        return str(self.__dict__)


class Playback(object):
    """
    Playback class creates an illumination instance that emulates how a hologram will played back. Therefore, it needs
    only one light source.

    :param source: an instance of Source subclasses.
    """

    def __init__(self, source):
        self.source = source

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        assert isinstance(value, Source), 'source must be an instance of {} class or its subclasses.'.format(
            Source)
        self._source = value


class TargetPlayback(Record):
    """
    TargetPlayback class creates an illumination instance that emulates how a hologram should playback in an attempt to
    design a recording process. Therefore, it needs two light sources. Due to the same requirement as the Record class,
    it is convenient to make TargetPlayback a subclass of Record.

    :param source1: an instance of Source subclasses.
    :param source2: an instance of Source subclasses.
    """

    def __init__(self, source1, source2):
        super(TargetPlayback, self).__init__(source1, source2)
