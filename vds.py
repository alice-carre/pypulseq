"""
# Program translated from the matlab Program of Brian Hargreaves:
http://mrsrl.stanford.edu/~brian/vdspiral/
#Adaptation was also made from the original program to change the default unit of the matlab program
#and were corrected in the description of the program
#In this new version the smax and gmax values are given in Hz/m/s and Hz/m/s and already consider the gamma factor

%	function [k,g,s,time,r,theta] = vds(smax,gmax,T,N,Fcoeff,rmax)
%
%	VARIABLE DENSITY SPIRAL GENERATION:
%	----------------------------------
%
%	Function generates variable density spiral which traces
%	out the trajectory
%
%			k(t) = r(t) exp(i*q(t)), 		[1]
%
%	Where q is the same as theta...
%		r and q are chosen to satisfy:
%
%		1) Maximum gradient amplitudes and slew rates.
%		2) Maximum gradient due to FOV, where FOV can
%		   vary with k-space radius r/rmax, as
%
%			FOV(r) = Sum    Fcoeff(k)*(r/rmax)^(k-1)   [2]
%
%
%	INPUTS:
%	-------
%	smax = maximum slew rate in Hz/m/s
%	gmax = maximum gradient in Hz/m (limited by Gmax or FOV)
%	T = sampling period (s) for gradient AND acquisition.
%	N = number of interleaves.
%	Fcoeff = FOV coefficients with respect to r - see above.
%	rmax= value of k-space radius at which to stop (m^-1).
%		rmax = 1/(2*resolution)
%
%
%	OUTPUTS:
%	--------
%	k = k-space trajectory (kx+iky) in m-1.
%	g = gradient waveform (Gx+iGy) in Hz/m.
%	s = derivative of g (Sx+iSy) in Hz/m/s.
%	time = time points corresponding to above (s).
%	r = k-space radius vs time (used to design spiral)
%	theta = atan2(ky,kx) = k-space angle vs time.
%
%
%	METHODS:
%	--------
%	Let r1 and r2 be the first derivatives of r in [1].
%	Let q1 and q2 be the first derivatives of theta in [1].
%	Also, r0 = r, and q0 = theta - sometimes both are used.
%	F = F(r) defined by Fcoeff.
%
%	Differentiating [1], we can get G = a(r0,r1,q0,q1,F)
%	and differentiating again, we get S = b(r0,r1,r2,q0,q1,q2,F)
%
%	(functions a() and b() are reasonably easy to obtain.)
%
%	FOV limits put a constraint between r and q:
%
%		dr/dq = N/(2*pi*F)				[3]
%
%	We can use [3] and the chain rule to give
%
%		q1 = 2*pi*F/N * r1				[4]
%
%	and
%
%		q2 = 2*pi/N*dF/dr*r1^2 + 2*pi*F/N*r2		[5]
%
%
%
%	Now using [4] and [5], we can substitute for q1 and q2
%	in functions a() and b(), giving
%
%		G = c(r0,r1,F)
%	and 	S = d(r0,r1,r2,F,dF/dr)
%
%
%	Using the fact that the spiral should be either limited
%	by amplitude (Gradient or FOV limit) or slew rate, we can
%	solve
%		|c(r0,r1,F)| = |Gmax|  				[6]
%
%	analytically for r1, or
%
%	  	|d(r0,r1,r2,F,dF/dr)| = |Smax|	 		[7]
%
%	analytically for r2.
%
%	[7] is a quadratic equation in r2.  The smaller of the
%	roots is taken, and the np.real part of the root is used to
%	avoid possible numeric errors - the roots should be np.real
%	always.
%
%	The choice of whether or not to use [6] or [7], and the
%	solving for r2 or r1 is done by findq2r2 - in this .m file.
%
%	Once the second derivative of theta(q) or r is obtained,
%	it can be integrated to give q1 and r1, and then integrated
%	again to give q and r.  The gradient waveforms follow from
%	q and r.
%
%	Brian Hargreaves -- Sept 2000.
%
%	See Brian's journal, Vol 6, P.24.
%
%
%	See also:  vds2.m,  vdsmex.m,  vds.c
%

% =============== CVS Log Messages ==========================
%	$Log: vds.m,v $
%	Revision 1.5  2004/04/27 18:08:44  brian
%	Changed FOV to a polynomial of unlimited length,
%	and hopefully changed all comments accordingly.
%	Also moved sub-functions into vds.m so that
%	no other .m files are needed.
%
%	Revision 1.4  2003/09/16 02:55:52  brian
%	minor edits
%
%	Revision 1.3  2002/11/18 05:36:02  brian
%	Rounds lengths to a multiple of 4 to avoid
%	frame size issues later on.
%
%	Revision 1.2  2002/11/18 05:32:19  brian
%	minor edits
%
%	Revision 1.1  2002/03/28 01:03:20  bah
%	Added to CVS
%
%
% ===========================================================
"""
import numpy as np
from matplotlib import pyplot as plt

def qdf(a:float, b:float, c:float) -> tuple[float,float]:

    d = b**2-4*a*c
    roots = ((-b+np.sqrt(d))/(2*a),(-b-np.sqrt(d))/(2*a))

    return roots

def findq2r2(smax:float, gmax:float, r:float, r1:float, T:float, Ts:float, N:int, Fcoeff:list, rmax:float) -> tuple[float,float]:

    F = 0 #FOV function value for this r.
    dFdr = 0 # dFOV/dr for this value of r.

    for rind in range(len(Fcoeff)):
        F += Fcoeff[rind]*(r/rmax)**rind
        if rind > 0:
            dFdr += rind*Fcoeff[rind]*(r/rmax)**(rind-1)/rmax

    GmaxFOV = 1 / F / Ts
    Gmax = min(GmaxFOV, gmax)

    maxr1 = np.sqrt(Gmax**2/(1+(2*np.pi*F*r/N)**2))
    if r1 > maxr1:
        #Grad amplitude limited.  Here we just run r upward as much as we can without
        #going over the max gradient.
        r2 = (maxr1-r1)/T
    else:
        twopiFoN = 2*np.pi*F/N
        twopiFoN2 = twopiFoN**2
        #A, B, C are coefficents of the equation which equates the slew rate
        #calculated from r, r1, r2 with the maximum gradient slew rate.
        # A * r2 * r2 + B * r2 + C = 0
        # A, B, C are in terms of F, dF / dr, r, r1, N and smax.
        A = r*r*twopiFoN2+1
        B = 2*twopiFoN2*r*r1*r1 + 2*twopiFoN2/F*dFdr*r*r*r1*r1
        C = twopiFoN2**2*r*r*r1**4 + 4*twopiFoN2*r1**4 + (2*np.pi/N*dFdr)**2*r*r*r1**4 + 4*twopiFoN2/F*dFdr*r*r1**4 \
            - smax**2
        rts = qdf(A, B, C)
        r2 = np.real(rts[0])
        slew = (r2 - twopiFoN2 * r * r1 ** 2 + 1j * twopiFoN * (2 * r1 ** 2 + r * r2 + dFdr / F * r * r1 ** 2))
        sr = np.abs(slew)/smax

        if np.abs(slew)/smax>1.01:
            tt = print('Slew violation, slew = ', round(abs(slew)), ' smax= ', round(smax), ' sr=', sr, ' r=', r, ' r1=', r1)

    q2 = 2 * np.pi / N * dFdr * r1**2 + 2 * np.pi * F / N * r2

    return q2, r2

def vds(smax:float, gmax:float, T:float, N:int, Fcoeff:list, rmax:float)-> tuple[np.array, np.array, np.array,
                                                                                 np.array, np.array, np.array]:

    oversampling = 8 # Keep this even.
    To = T/oversampling # To is the period with oversampling.

    q0=0
    q1=0
    r0=0
    r1=0

    t=0
    count=0

    theta = np.zeros((1000000,1))
    r = np.zeros((1000000,1))
    time = np.zeros((1000000,1))

    while r0 < rmax:
        q2, r2 = findq2r2(smax, gmax, r0, r1, To, T, N, Fcoeff, rmax)

        #Integrate for r, r', theta and theta'
        q1= q1 + q2 * To
        q0 = q0 + q1 * To
        t += To

        r1 += r2*To
        r0 += r1 * To

        #Store
        count += 1
        theta[count] = q0
        r[count] = r0
        time[count] = t

        if (count % 100) == 0:
             tt = print(count,' points',' |k|=', r0)

    r = r[round(oversampling/2):count:oversampling]
    theta = theta[round(oversampling/2):count:oversampling]
    time = time[round(oversampling/2):count:oversampling]

    #Keep the length a multiple of 4, to save pain..!
    ltheta = 4 * int(np.floor(np.shape(theta)[0] / 4))
    r = r[0:ltheta]
    theta = theta[0:ltheta]
    time = time[0:ltheta]

    k = r * np.exp(1j * theta)
    k1 = np.zeros((np.shape(k)[0]+1, 1))*1j
    k2 = np.zeros((np.shape(k)[0]+1, 1))*1j
    k1[1:] = k
    k2[:-1] = k

    g = (k1 - k2) / T
    g = g[0:np.shape(k)[0]]

    s1 = np.zeros((np.shape(k)[0]+1, 1))*1j
    s2 = np.zeros((np.shape(k)[0]+1, 1))*1j
    s1[1:] = g
    s2[:-1] = g
    s = (s1 - s2) / T
    s = s[0:len(k)]

    """
    #Plot
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot( np.real(k), np.imag(k))
    plt.title('k_y vs k_x')
    #axis('square')

    plt.subplot(2, 2, 2)
    plt.plot(time, np.real(k), 'r--', time, np.imag(k), 'b--', time, abs(k), 'k-')
    plt.title('k vs t') 
    plt.ylabel('k (m^{-1})') 

    plt.subplot(2, 2, 3)
    plt.plot(time, np.real(g), 'r--', time, np.imag(g), 'b--', time, abs(g), 'k-');
    plt.title('g vs t')
    plt.ylabel('G (Hz/m)')

    plt.subplot(2, 2, 4) 
    plt.plot(time, np.real(s), 'r--', time, np.imag(s), 'b--', time, abs(s), 'k-') 
    plt.title('s vs t')
    plt.ylabel('Slew Rate (Hz/m/s)')

    plt.show()
    """

    return k.flatten(),g.flatten(),s.flatten(),time.flatten(),r.flatten(),theta.flatten()
