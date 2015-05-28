#!/usr/bin/env python
import numpy as np

""" STM simulation based on Tersoff-Hamann Theory"""

class stm:
    def __init__(self, inFile='PARCHG'):
        self.inFile = inFile

        self.cell = None
        self.a = None
        self.b = None
        self.c = None
        self.position = None
        self.zmaxp = None
        self.zmaxi = None

        self.nx = None
        self.ny = None
        self.nz = None
        self.nxyz = None
        self.chg = None

        self.elenums = []
        self.element = []
        self.symbols = []
        
        self.readChg()

        self.Kinds = len(self.element)
        self.Natoms = self.elenums.sum()
        self.AllowedBond = []
        self.BondMaxL = []
        self.bonds = []

    def readChg(self):
        """ Read VASP CHGCAR-like file """

        chgcar = [line.strip() for line in open(self.inFile, 'r')]

        isep = chgcar.index('')

        pos = chgcar[:isep]
        self.nx, self.ny, self.nz = [int(x) for x in chgcar[isep+1].split()]
        self.nxyz = self.nx * self.ny * self.nz

        # the charge part of PARCHG
        chg = ' '.join(chgcar[isep+2:]).split()[:self.nxyz]
        self.chg = np.array(chg, dtype=float).reshape((self.nz, self.ny, self.nx))

        # the position part of PARCHG
        scale = float(pos[1])
        self.cell = scale * np.array([line.split() for line in pos[2:5]], dtype=float)
        position = np.array([line.split() for line in pos[8:]], dtype=float)

        self.position = np.dot(self.cell, position.T).T

        self.element = pos[5].split()
        if not self.element[0].isalpha():
            raise RuntimeError('Format not VASP 5.X')
        self.elenums = np.array([int(x) for x in pos[6].split()], dtype=int)
        assert len(self.elenums) == len(self.element)

        self.symbols = [it for n,s in zip(self.elenums, self.element) 
                           for it in ((' ' + s + ' ') * n).split()]

        # basis vector and length of basis
        self.a, self.b, self.c = [np.linalg.norm(x) for x in self.cell]

        # maximum height of the slab
        self.zmaxp = self.position[:,2].max()
        self.zmaxi = int(self.zmaxp / self.c * self.nz)
        #

    def current(self, zcut=None, repeat=(2,2), ext=0.10, save=False):
        """ gei iso-height image """

        if not zcut:
            zcut = int((self.zmaxp + 2.0) / self.c * self.nz)
        zext = int(self.nz * ext)

        # average current
        c = np.average(self.chg[zcut,:,:])
        # height of iso-current 
        img = np.argmin(np.abs(self.chg[zcut-zext:zcut+zext, :,:] - c), axis=0)

        img_ext = np.tile(img, repeat[::-1]) + zcut - zext
        exts = (0, self.a * repeat[1], 0, self.b * repeat[0])

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(img_ext, extent=exts,
                    origin='lower',
                    cmap='hot', interpolation='bicubic')
            plt.xticks([])
            plt.yticks([])

            for vx, vy in self.bonds:
                vi = vx.copy()
                vj = vy.copy()

                vm = (vi + vj) / 2.
                vm[vm < 0] = 0
                if vm[0] > self.a:
                    vm[0] = self.a
                if vm[1] > self.b:
                    vm[1] = self.b
                
                if 0 < vi[0] < self.a and 0 < vi[1] < self.b:
                    ax.plot([vi[0], vm[0]], [vi[1], vm[1]],
                            alpha=0.6,
                            lw=1.5, color='white')
                if 0 < vj[0] < self.a and 0 < vj[1] < self.b:
                    ax.plot([vj[0], vm[0]], [vj[1], vm[1]],
                            alpha=0.6,
                            lw=1.5, color='white')
            if save:
                np.save('current_%d.npy' % zcut, img_ext)
                plt.savefig('current_%d.png' % zcut, dpi=240)
            plt.show()
        except:
            pass

        return c, img_ext

    def height(self, zcut=None, repeat=(2,2)):
        """ gei iso-height image """

        if not zcut:
            zcut = int((self.zmaxp + 2.0) / self.c * self.nz)

        # a cut plane through z = zcut
        img_ext = np.tile(self.chg[zcut, :,:], repeat[::-1])
        exts = (0, self.a * repeat[1], 0, self.b * repeat[0])

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plt.imshow(img_ext, extent=exts,
                    origin='lower',
                    cmap='hot', interpolation='bicubic')
            plt.xticks([])
            plt.yticks([])
            if save:
                np.save('current_%d.npy' % zcut, img_ext)
                plt.savefig('current_%d.png' % zcut, dpi=240)
            plt.show()
        except:
            pass

        return img_ext

    def get_average_current(self):
        cc = np.zeros(self.nz)

        for i in range(self.nz):
            cc[i] = np.average(self.chg[i,:,:])

        return cc

    def creatBond(self, zcmax=None, zcmin=0.0):
        """ Create bond list of the system """

        N = self.Natoms
        ij = [(i,j) for i in range(N) for j in range(N) if i > j]

        ele = [x for x in set(self.element)]
        self.AllowedBond = [set((ele[i], ele[j]))
                            for i in range(len(ele))
                            for j in range(len(ele))
                            if i > j]
        # print self.AllowedBond

        if zcmax:
            try:
                zcmax = float(zcmax)
            except TypeError:
                raise

        if not self.BondMaxL:
            print "Please enter the Maximum Distance between a pair of atoms!"
            print "enter -1 if the pair should NOT be connected!"
            print 
            for x, y in self.AllowedBond:
                L = raw_input("(%3s %3s): " % (x, y))
                self.BondMaxL += [float(L)]
        
        self.bonds = []
        for i, j in ij:
            vi = self.position[i,:]
            vj = self.position[j,:]
            if zcmax:
                if not (zcmin < vi[2] < zcmax or zcmin < vj[2] < zcmax):
                    continue

            sym_ij = set((self.symbols[i], self.symbols[j]))

            if sym_ij in self.AllowedBond:
                ii = self.AllowedBond.index(sym_ij)
                L = self.BondMaxL[ii]

                for ix in range(-1,2):
                    for iy in range(-1,2):
                        va = vi + ix * self.cell[0,:] + iy * self.cell[1,:]
                        vb = vj
                        vab = np.sqrt(np.sum((va - vb)**2))
                        # print va, vb
                        if vab <= L:
                            self.bonds += [(va, vb)]

                for ix in range(-1,2):
                    for iy in range(-1,2):
                        va = vj + ix * self.cell[0,:] + iy * self.cell[1,:]
                        vb = vi
                        vab = np.sqrt(np.sum((va - vb)**2))
                        if vab <= L and not ix == iy == 0:
                            self.bonds += [(va, vb)]


r0 = stm('stat0/stm/PARCHG.10')
r0.BondMaxL = [1.0, -1.0, 2.3]
r0.creatBond(20, 16)
r0.current(230, repeat=(1,1))

