"""Unit conversion constants and helpers."""

ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

FS_TO_AU_TIME = 41.341374575751
AU_TIME_TO_FS = 1.0 / FS_TO_AU_TIME

AMU_TO_ELECTRON_MASS = 1822.888486209
ELECTRON_MASS_TO_AMU = 1.0 / AMU_TO_ELECTRON_MASS

EV_TO_HARTREE = 0.0367493
HARTREE_TO_EV = 1.0 / EV_TO_HARTREE
HARTREE_TO_CM1 = 219474.6313705
CM1_TO_HARTREE = 1.0 / HARTREE_TO_CM1

KB_HARTREE_PER_K = 3.1668115634556e-6


def angstrom_to_bohr(value):
    return value * ANGSTROM_TO_BOHR


def bohr_to_angstrom(value):
    return value * BOHR_TO_ANGSTROM


def fs_to_au_time(value):
    return value * FS_TO_AU_TIME


def au_time_to_fs(value):
    return value * AU_TIME_TO_FS


def amu_to_me(value):
    return value * AMU_TO_ELECTRON_MASS


def me_to_amu(value):
    return value * ELECTRON_MASS_TO_AMU


def kelvin_to_hartree(value):
    return value * KB_HARTREE_PER_K


def hartree_to_kelvin(value):
    return value / KB_HARTREE_PER_K


def hartree_to_cm1(value):
    return value * HARTREE_TO_CM1


def cm1_to_hartree(value):
    return value * CM1_TO_HARTREE
