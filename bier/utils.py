# -*- coding: utf-8 -*-
#    This program is part of BIER (Basic Irradiance Estimation and Regression).
#    Copyright (C) 2016 Tobias Kölling <tobias.koelling@physik.uni-muenchen.de>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Collection of utility functions for BIER.
"""


def lazyprop(func):
    """
    creates a property which is only evaluated once and cached afterwards
    """
    attr_name = '_lazy_' + func.__name__

    @property
    def _lazyprop(self):
        """ wrapper for cacheing logic """
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazyprop


def split_envi_blocks(lines):
    """
    Splits a line-separated iterable into toplevel ENVI-key-value blocks.
    """
    opened = 0
    current_block = ""
    for line_no, line in enumerate(lines):
        if line[0] == ';':
            continue
        line = line.strip()
        current_block += line
        opened += line.count('{') - line.count('}')
        if opened == 0:
            yield (line_no, current_block)
            current_block = ""
    if opened != 0:
        raise ValueError('missmatched parantheses at EOF '
                         '(still expecting %d closing brackets)' % opened)


def split_paren(text, split_char=',', opening='{', closing='}'):
    """
    splits a string only outside of parenthesis
    """
    opened = 0
    splitpoints = []
    for i, char in enumerate(text):
        if char == opening:
            opened += 1
        elif char == closing:
            opened -= 1
        elif char == split_char and opened == 0:
            splitpoints.append(i)
    for begin, end in zip([-1] + splitpoints, splitpoints + [len(text)]):
        yield text[begin+1:end]


def parse_envi_value(value):
    """
    parses a value of an envi header into either a string or a list
    """
    value = value.strip()
    if len(value) == 0:
        # this is an empty string
        return ''
    elif value[0] == '{':
        # this is a list
        return map(parse_envi_value, split_paren(value[1:-1]))
    else:
        # this is a string
        return value


def read_envi_header(filename, graceful=False):
    """
    USAGE: hdr = readEnviHeader(file)

    Reads an ENVI ".hdr" file header and returns the parameters in
    a dictionary as strings.
    """
    out = {}
    with open(filename) as filehandle:
        try:
            magic = filehandle.next().strip()
        except StopIteration:
            raise ValueError('header is empty')
        if magic != 'ENVI':
            raise ValueError('not an ENVI header')
        for line_no, tlb in split_envi_blocks(filehandle):
            try:
                name, value = tlb.split('=', 1)
            except ValueError:
                if tlb == '':
                    continue
                else:
                    if graceful:
                        continue
                    else:
                        raise ValueError('ENVI file parse error in: %s:%d' %
                                         (filename, line_no+2))
            out[name.strip()] = parse_envi_value(value)
    # sometimes description is written list-like but is expected to be one text
    if 'description' in out and isinstance(out['description'], list):
        out['description'] = ','.join(out['description'])
    dot_items = {key: value for key, value in out.items() if '.' in key}
    for key in dot_items:
        del out[key]
    for key, value in dot_items.items():
        keys = key.split('.')
        cur = out
        while len(keys) > 1:
            try:
                cur = cur[keys[0]]
            except KeyError:
                cur[keys[0]] = {}
                cur = cur[keys[0]]
            keys = keys[1:]
        cur[keys[0]] = value
    return out


def parse_iso_date(date):
    """
    Convert an iso formatted datetime string to datetime object
    """
    import dateutil.parser as dateparser
    import dateutil.tz as datetz
    date = dateparser.parse(date)
    try:
        date = date.astimezone(datetz.tzutc())  # pylint: disable=no-member
    except ValueError:
        date.replace(tzinfo=datetz.tzutc())  # pylint: disable=no-member
    return date
