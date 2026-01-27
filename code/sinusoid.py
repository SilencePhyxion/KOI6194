from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np


class Sinusoid(object):
    """
    A single sinusoid.

    The sinusoid is of the form: $A\sin(2\pi f t + \delta)$, where $A$ is the
    amplitude, $f$ is the frequency (*not* the angular frequency) and
    $\delta$ is the phase.

    Parameters
    ----------
    frequency : float
        Frequency (*not* angular frequency) of the sinusoid.
    amplitude : float
        Semi-amplitude of the sinusoid.
    delta : float
        Phase of the sinusoid (radians).

    Attributes
    ----------
    angular_frequency : float
        Angular frequency of the sinusoid.
    """
    def __init__(self, frequency, amplitude, phase):
        self.amplitude = amplitude
        self.phase = phase
        self.frequency = frequency
        self.angular_frequency = 2 * np.pi * self.frequency

    def __repr__(self):
        return "{} sin(2pi {} t + {})".format(self.amplitude,
                                              self.frequency, self.phase)

    def __call__(self, t):
        return self.amplitude * np.sin(self.angular_frequency * t + self.phase)


class SinusoidModel(object):
    """
    Represent a combination of sinusoids

    Every model contains, at a minimum, a zero-frequency (i.e. DC) mode.

    >>> from sinusoid import SinusoidModel
    >>> model = SinusoidModel()
    >>> f1 = 1.1    #  frequency, NOT angular frequency
    >>> f2 = 1.7
    >>> f3 = 3.5
    >>> model.frequencies =[f1, f2, f3] # "base" frequencies -- NOT angular
    >>> model.modes = [[1,0,0],     # first mode has frequency f1;
                        [1, 0, 2]]  # second mode has frequency f1 + 2*f2
    >>> model._fit_parameters = [0.5, 1., # 1st mode amplitude 0.5, phase 1 rad
                                1.0, 0]   # 2nd mode amplitude 1.0, phase 0 rad
    """
    def __init__(self, frequencies=None, modes=None, amp_phase=None):
        self._sinusoids = []
        self._frequencies = ()
        self._modes = ()
        self.frequencies = (frequencies or [])
        self.modes = (modes or [])
        self.dc_offset = 0
        if amp_phase:
            self._fit_parameters = amp_phase

    def __repr__(self):
        default_format = "{:^{width}}{:^{width}}{:^{width}}{:^{width}}"
        width = 20
        output = [default_format.format("Mode", "Frequency", "Amplitude",
                                        "Phase", width=width)]
        dashes = (width - 4) * "-"
        output.append(default_format.format(dashes, dashes, dashes, dashes,
                                            width=width))
        DC = default_format.format("DC", "--", self.dc_offset,
                                   "--", width=width)
        output.append(DC)
        sinusoids = [default_format.format(self._pretty_mode(mode),
                                           sinusoid.frequency,
                                           sinusoid.amplitude,
                                           sinusoid.phase,
                                           width=width)
                     for mode, sinusoid in zip(self.modes, self._sinusoids)]
        output.extend(sinusoids)
        return "\n".join(output)

    def __call__(self, t):
        v = 0
        for sinusoid in self._sinusoids:
            v += sinusoid(t)

        v += self.dc_offset

        return v

    def _pretty_mode(self, mode):
        """
        Generate nice looking string for mode
        """
        mode_str = []
        for freq_num, weight in enumerate(mode):
            if weight != 0:
                if weight == 1:
                    mode_str.append("+f{}".format(freq_num))
                elif weight == -1:
                    mode_str.append("-f{}".format(freq_num))
                else:
                    mode_str.append("{:+}f{}".format(weight, freq_num))
        return "".join(mode_str)

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, freq):

        if len(set(freq)) != len(freq):
            raise ValueError('One or more repeated frequencies provided in input')

        self._validate_frequencies(freq)

        n_new_frequencies = len(freq)
        n_current_frequencies = len(self.frequencies)

        if (n_new_frequencies < n_current_frequencies):
            raise ValueError("Cannot decrease number of frequencies. Define a new model instead.")

        if ((n_current_frequencies > 0) and
            (n_new_frequencies > n_current_frequencies)):
            raise ValueError("Use method add_frequency to increase the number of frequencies")

        try:
            self._frequencies = tuple([f for f in freq])
        except TypeError:
            self._frequencies = (freq)

    def add_frequency(self, *frequencies, **kwd):
        """
        Add one or more frequencies to model

        Parameters
        ----------

        frequencies : float
            One or more frequencies to be appended to the model

        extend_modes : boolean, optional
            Controls whether adding a frequency automatically extends existing
            modes by appending zeros. Default is True. If set to False
            current modes are deleted.
        """

        self._validate_frequencies(frequencies)

        all_frequencies = list(self.frequencies)
        all_frequencies.extend(frequencies)

        extend_modes = kwd.pop('extend_modes', True)

        if not extend_modes:
            self.modes = []
            return

        zeros = [0 for frequency in frequencies]
        modes = []
        self._frequencies = tuple(all_frequencies)
        for mode in self.modes:
            new_mode = list(mode)
            new_mode.extend(zeros)
            modes.append(new_mode)
        self.modes = modes

    def _validate_frequencies(self, new_frequencies):
        """
        Check whether new frequencies duplicate any existing frequencies.
        """

        overlap = set(new_frequencies).intersection(set(self.frequencies))
        if overlap:
            bad_frequencies = ', '.join([str(i) for i in overlap])
            raise ValueError('The frequencies ' + bad_frequencies +
                             ' are already in the model')

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, mode_list):
        self._modes = []
        self._sinusoids = []

        for mode in (mode_list or ()):
            if len(mode) != len(self.frequencies):
                raise ValueError('Wrong number of modes in mode setter' +
                                 ' for mode {}'.format(mode))
            if tuple(mode) in self.modes:
                raise ValueError('Mode {0} is already in the model; model is unchanged'.format(str(tuple(mode))))
            frequency = (np.array(self.frequencies) * np.array(mode)).sum()
            if not (frequency > 0):
                raise ValueError('Zero frequency mode given.')
            self._sinusoids.append(Sinusoid(frequency, 0., 0.))
            self._modes.append(tuple(mode))
        self._modes = tuple(self._modes)

    def add_mode(self, *mode_description):
        """
        Add one or more modes to current model.

        Parameters
        ----------

        mode : floats or lists
            Two types of input are allowed, either a series of floats, one
            for each frequency in the model, or a series of lists, each of
            which defines a mode.

        The example below illustrates the two methods of calling `add_mode`.

        >>> from sinusoid import SinusoidModel
        >>> another_model = SinusoidModel()
        >>> another_model.add_frequency(1.2, 2.3)

        Add modes one at a time:

        >>> another_model.add_mode(1, 0)
        >>> another_model.modes
        ((1, 0), )

        Add two modes at once:
        >>> another_model.add_mode([0, 1], [1, 1])
        ((1, 0), (0, 1), (1, 1))

        """
        modes_to_add = []
        for mode in mode_description:
            modes_to_add.append(mode)

        try:
            modes_to_add[0][0]
        except TypeError:
            modes_to_add = [modes_to_add]
        new_mode_list = list(self.modes)
        new_mode_list.extend(modes_to_add) # type: ignore
        self.modes = new_mode_list

    @property
    def _fit_parameters(self):
        p = [self.dc_offset]
        for sinusoid in self._sinusoids:
            p.append(sinusoid.amplitude)
            p.append(sinusoid.phase)
        return p

    @_fit_parameters.setter
    def _fit_parameters(self, p):
        self.dc_offset = p[0]
        sines = p[1:]
        if  (len(sines) % 2) == 1:
            raise ValueError('Must supply an even number of fit parameters')
        for i in range(len(sines) // 2): # type: ignore
            self._sinusoids[i].amplitude = sines[2 * i]
            self._sinusoids[i].phase = sines[2 * i + 1]

    def value(self, t):
        v = 0
        for sinusoid in self._sinusoids:
            v += sinusoid(t)

        v += self.dc_offset

        return v

    def fit_to_data(self, time, data, initial_parameters=[]):
        from scipy import optimize

        if not initial_parameters:
            initial_parameters = 0 * np.array(self._fit_parameters) + 1.

        self._fit_parameters = initial_parameters

        def errfunc(p, model, t, dat):
            model._fit_parameters = p
            return model.value(t) - dat

        params, junk = optimize.leastsq(errfunc, initial_parameters,
                                        args=(self, time, data))

        self._fit_parameters = params

        for sinusoid in self._sinusoids:
            if sinusoid.amplitude < 0:
                sinusoid.amplitude *= -1
                sinusoid.phase += np.pi

            while sinusoid.phase > 2 * np.pi:
                sinusoid.phase -= 2 * np.pi

            while sinusoid.phase < 0:
                sinusoid.phase += 2 * np.pi
