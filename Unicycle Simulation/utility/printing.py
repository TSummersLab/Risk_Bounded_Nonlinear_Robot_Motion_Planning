"""Printing utilities"""
# Author: Ben Gravell

import sys


def inplace_print(printstr):
    delete = '\b' * len(printstr)
    print("{0}{1}".format(delete, printstr), end="")
    sys.stdout.flush()


def writeout(string, file, end=''):
    file.write(string + end)


def printout(string, file):
    print(string)
    writeout(string, file)


class printcolors:
    """
    Simple class that holds several ANSI escape sequences to enable colored/bold/underline printing.
    Taken from https://godoc.org/github.com/whitedevops/colors
    Remember to print(printcolors.Default) and/or print(printcolors.BackgroundDefault)
    to reset printing to default settings.
    """
    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"


def create_tag(message, message_type='info', prefix='--', suffix='', end=''):
    """
    Create a formatted tag string for diagnostic printing.
    """

    # Convert from message type to color/format specification
    if message_type == 'info':
        color_str = printcolors.LightBlue
    elif message_type == 'fail':
        color_str = printcolors.Red
    elif message_type == 'pass':
        color_str = printcolors.Green

    # Build the tag
    tag_str = ''
    tag_str += color_str
    tag_str += prefix
    tag_str += message
    tag_str += suffix
    tag_str += printcolors.Default
    tag_str += end

    return tag_str