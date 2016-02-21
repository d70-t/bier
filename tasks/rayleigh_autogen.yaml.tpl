constants:
    f_dd: 0
    f_ds: 1
    H_oz: 0.4
    alpha: 2.0
    wvl_a: 550
    RH: .6
    P: 1013.25
    AM: 1
    WV: 0
    beta: 0
    g_dsa: 0
    theta_sun: True
    el: True
    sca: True
    t: 0.144
    #g_dsr: 0.17
guess:
    beta: .13
    WV: .1
    g_dsr: .170
    g_dsa: .01
    el: .75
    sca: 0
    t: 0.14
    s: 0.0
lims:
    beta:  [0, 20]
    WV:    [0, 20]
    g_dsr: [0, 1000]
    g_dsa: [0, 1000]
    el:    [0, 1.571] #about pi/2
    sca:   [0, 6.2832] #about 2*pi
    t:     [-1, 1]
    s:     [-1, 1]
weights: ../weight.txt