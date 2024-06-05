'''

module for galaxy catalogs. Module includes functions to construct galaxy
catalogs using halo occupation distribution (HOD) models. HOD models paint
galaxies onto halo catalogs constructed from N-body simulations 



References
----------
* make_survey pipeline from BOSS: https://trac.sdss3.org/browser/repo/mockFactory/trunk/make_survey?order=name


'''
import os 
import numpy as np 
# --- nbodykit --- 
import nbodykit.lab as NBlab
from astropy.utils.misc import NumpyRNGContext
from nbodykit.hod import Zheng07Model, HODModel
from halotools.empirical_models import NFWPhaseSpace, BiasedNFWPhaseSpace
from halotools.empirical_models.phase_space_models import MonteCarloGalProf


def thetahod_literature(paper): 
    ''' best-fit HOD parameters from the literature. 
    
    Currently, HOD values from the following papers are available:
    * 'parejko2013_lowz'
    * 'manera2015_lowz_ngc'
    * 'manera2015_lowz_sgc'
    * 'redi2014_cmass'
    '''
    if paper == 'parejko2013_lowz': 
        # lowz catalog from Parejko+2013 Table 3. Note that the 
        # parameterization is slightly different so the numbers need to 
        # be converted.
        p_hod = {
                'logMmin': 13.25, 
                'sigma_logM': 0.43, #0.7 * sqrt(2) * log10(e)
                'logM0': 13.27, # log10(kappa * Mmin)
                'logM1': 14.18, 
                'alpha': 0.94
                } 
    elif paper == 'manera2015_lowz_ngc':
        # best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015) 
        p_hod = {
                'logMmin': 13.20, 
                'sigma_logM': 0.62, 
                'logM0': 13.24, 
                'logM1': 14.32, 
                'alpha': 0.9
                }
    elif paper == 'manera2015_lowz_sgc':
        # best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015) 
        # Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        # currently implemented is primarily for the 0.2 < z < 0.35 population, 
        # which has nbar~3x10^-4 h^3/Mpc^3
        p_hod = {
                'logMmin': 13.14, 
                'sigma_logM':0.55,
                'logM0': 13.43, 
                'logM1': 14.58, 
                'alpha': 0.93 
                }
    elif paper == 'reid2014_cmass': 
        # best-fit HOD from Reid et al. (2014) Table 4
        p_hod = {
                'logMmin': 13.03, 
                'sigma_logM':0.38,
                'logM0': 13.27, 
                'logM1': 14.08, 
                'alpha': 0.76
                }
    else:
        raise NotImplementedError
    
    return p_hod 


def hodGalaxies(halos, p_hod, seed=None, hod_model='zheng07'): 
    ''' populate given halo catalog (halos) with galaxies based on HOD model
    with p_hod parameters. Currently only supports the Zheng+(2007) model.

    Parameters
    ----------
    p_hod : dict
        dictionary specifying the HOD parameters 
    '''
    # check HOD parameters
    if 'alpha' not in p_hod.keys(): 
        raise ValueError
    if 'logMmin' not in p_hod.keys(): 
        raise ValueError
    if 'logM1' not in p_hod.keys(): 
        raise ValueError
    if 'logM0' not in p_hod.keys(): 
        raise ValueError
    if 'sigma_logM' not in p_hod.keys(): 
        raise ValueError

    # populate using HOD
    if hod_model == 'zheng07': 
        hod = halos.populate(Zheng07Model, seed=seed, **p_hod) 
    elif hod_model == 'zheng07_ab': 
        hod = halos.populate(AssembiasZheng07Model, seed=seed, **p_hod) 
    else: 
        raise NotImplementedError
    return hod 


def BOSSGalaxies(sample='lowz-south'): 
    ''' Read in BOSS galaxy catalog. Data can be downloaded from 
    https://data.sdss.org/sas/dr12/boss/lss/
    

    Parameters
    ----------
    sample : string
        Specify the exact BOSS sample. For options see
        https://data.sdss.org/sas/dr12/boss/lss/
        (Default: 'cmass-north')

    Returns
    -------
    data : nbodykit.lab.FITSCatalog object
        BOSS galaxy catalog  
    '''
    if sample == 'lowz-south': 
        fgal = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat',
                'galaxy_DR12v5_LOWZ_South.fits.gz')
    elif sample == 'cmass-south': 
        fgal = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat',
                'galaxy_DR12v5_CMASS_South.fits.gz')
    elif sample == 'cmass-north': 
        fgal = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat',
                'galaxy_DR12v5_CMASS_North.fits.gz')
    else: 
        raise NotImplementedError

    data = NBlab.FITSCatalog(fgal)
    return data


class AssembiasZheng07Model(HODModel):
    @staticmethod
    def to_halotools(cosmo, redshift, mdef, concentration_key=None, **kwargs):
        """
            Return the Zheng 07 HOD model decorated with assembly bias.

        Parameters
        ----------
        cosmo :
            the nbodykit or astropy Cosmology object to use in the model
        redshift : float
            the desired redshift of the model
        mdef : str, optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or
            'XXXm' where 'XXX' is an int specifying the overdensity
        concentration_key : str
            the name of the column that will specify concentration; if not
            provided, the analytic formula from
            `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_
            is used.
        **kwargs :
            additional keywords passed to the model components; see the
            Halotools documentation for further details
        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        from nbodykit.cosmology import Cosmology
        from halotools.empirical_models import AssembiasZheng07Sats, AssembiasZheng07Cens, NFWPhaseSpace, TrivialPhaseSpace
        from halotools.empirical_models import HodModelFactory 

        kwargs.setdefault('modulate_with_cenocc', True)

        # need astropy Cosmology
        if isinstance(cosmo, Cosmology):
            cosmo = cosmo.to_astropy()

        # determine concentration key
        if concentration_key is None:
            conc_mass_model = 'dutton_maccio14'
        else:
            conc_mass_model = 'direct_from_halo_catalog'

        # determine mass column
        mass_key = 'halo_m' + mdef

        # occupation functions
        cenocc = AssembiasZheng07Cens(prim_haloprop_key=mass_key, **kwargs)
        satocc = AssembiasZheng07Sats(prim_haloprop_key=mass_key, cenocc_model=cenocc, **kwargs)
        satocc._suppress_repeated_param_warning = True

        # profile functions
        kwargs.update({'cosmology':cosmo, 'redshift':redshift, 'mdef':mdef})
        censprof = TrivialPhaseSpace(**kwargs)
        satsprof = BiasedNFWPhaseSpace(conc_mass_model=conc_mass_model, **kwargs)

        # make the model
        model = {}
        model['centrals_occupation'] = cenocc
        model['centrals_profile'] = censprof
        model['satellites_occupation'] = satocc
        model['satellites_profile'] = satsprof
        return HodModelFactory(**model)


class VelAssembiasZheng07Model(HODModel):
    @staticmethod
    def to_halotools(cosmo, redshift, mdef, concentration_key=None, **kwargs):
        """
            Return the Zheng 07 HOD model decorated with assembly bias.

        Parameters
        ----------
        cosmo :
            the nbodykit or astropy Cosmology object to use in the model
        redshift : float
            the desired redshift of the model
        mdef : str, optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or
            'XXXm' where 'XXX' is an int specifying the overdensity
        concentration_key : str
            the name of the column that will specify concentration; if not
            provided, the analytic formula from
            `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_
            is used.
        **kwargs :
            additional keywords passed to the model components; see the
            Halotools documentation for further details
        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        from nbodykit.cosmology import Cosmology
        from halotools.empirical_models import AssembiasZheng07Sats, AssembiasZheng07Cens
        from halotools.empirical_models import HodModelFactory 

        kwargs.setdefault('modulate_with_cenocc', True)

        # need astropy Cosmology
        if isinstance(cosmo, Cosmology):
            cosmo = cosmo.to_astropy()

        # determine concentration key
        if concentration_key is None:
            conc_mass_model = 'dutton_maccio14'
        else:
            conc_mass_model = 'direct_from_halo_catalog'

        # determine mass column
        mass_key = 'halo_m' + mdef

        # occupation functions
        cenocc = AssembiasZheng07Cens(prim_haloprop_key=mass_key, **kwargs)
        satocc = AssembiasZheng07Sats(prim_haloprop_key=mass_key, cenocc_model=cenocc, **kwargs)
        satocc._suppress_repeated_param_warning = True

        # profile functions
        kwargs.update({'cosmology':cosmo, 'redshift':redshift, 'mdef':mdef})
        censprof = Centrals_vBiasedNFWPhaseSpace(conc_mass_model=conc_mass_model, **kwargs)
        satsprof = Satellites_vBiasedNFWPhaseSpace(conc_mass_model=conc_mass_model, 
                conc_gal_bias_bins=np.linspace(0.1, 2, 96), **kwargs)

        # make the model
        model = {}
        model['centrals_occupation'] = cenocc
        model['centrals_profile'] = censprof
        model['satellites_occupation'] = satocc
        model['satellites_profile'] = satsprof
        return HodModelFactory(**model)


class Centrals_vBiasedNFWPhaseSpace(NFWPhaseSpace): 
    ''' Model for the phase space distribution of galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995), where
    
    * the concentration of the tracers is permitted to differ from the
    host halo concentration 

    * the velocity dispersion of the tracers is permitted to differ from
    the host halo velocity dispersion 
    '''
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.param_dict = {'eta_vb.centrals': 0.0} 

    def assign_phase_space(self, table, seed=None):
        r""" Primary method of the `NFWPhaseSpace` class
        called during the mock-population sequence.

        Parameters
        -----------
        table : object
            `~astropy.table.Table` storing halo catalog.

            After calling the `assign_phase_space` method,
            the `x`, `y`, `z`, `vx`, `vy`, and `vz`
            columns of the input ``table`` will be over-written
            with their host-centric values.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        """
        phase_space_keys = ["x", "y", "z"]
        for key in phase_space_keys:
            table[key][:] = table["halo_" + key][:]

        if seed is not None:
            seed += 1
        MonteCarloGalProf.mc_vel(self, table, seed=seed)

    def mc_radial_velocity(self, scaled_radius, total_mass, *profile_params, **kwargs):
        r"""
        Method returns a Monte Carlo realization of radial velocities drawn from Gaussians
        with a width determined by the solution to the isotropic Jeans equation.
        Parameters
        ----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.
        total_mass: array_like
            Length-Ngals numpy array storing the halo mass in :math:`M_{\odot}/h`.
        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s).
            In the simplest case, this sequence has a single element,
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies.
            More generally, there should be a ``profile_params`` sequence item for
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.gal_prof_param_keys``.
        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
        Returns
        -------
        radial_velocities : array_like
            Array of radial velocities drawn from Gaussians with a width determined by the
            solution to the isotropic Jeans equation.
        """
        virial_velocities = self.virial_velocity(total_mass)
        radial_dispersions = virial_velocities * self.param_dict['eta_vb.centrals'] 

        seed = kwargs.get('seed', None)
        with NumpyRNGContext(seed):
            radial_velocities = np.random.normal(scale=radial_dispersions)

        return radial_velocities


class Satellites_vBiasedNFWPhaseSpace(BiasedNFWPhaseSpace): 
    ''' Model for the phase space distribution of galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995), where
    
    * the concentration of the tracers is permitted to differ from the
    host halo concentration 

    * the velocity dispersion of the tracers is permitted to differ from
    the host halo velocity dispersion 
    '''
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def _initialize_conc_bias_param_dict(self, **kwargs):
        r""" Set up the appropriate number of keys in the parameter dictionary
        and give the keys standardized names.
        """
        if 'conc_gal_bias_logM_abscissa' in list(kwargs.keys()):
            _conc_bias_logM_abscissa = np.atleast_1d(
                kwargs.get('conc_gal_bias_logM_abscissa')).astype('f4')
            d = ({'conc_gal_bias_param'+str(i): 1.
                for i in range(len(_conc_bias_logM_abscissa))})
            d2 = ({'conc_gal_bias_logM_abscissa_param'+str(i): float(logM)
                for i, logM in enumerate(_conc_bias_logM_abscissa)})
            self._num_conc_bias_params = len(_conc_bias_logM_abscissa)
            for key, value in d2.items():
                d[key] = value
            return d

        else:
            return {'conc_gal_bias.satellites': 1., 'eta_vb.satellites': 1.}

    def assign_phase_space(self, table, seed=None):
        r""" Primary method of the `NFWPhaseSpace` class
        called during the mock-population sequence.

        Parameters
        -----------
        table : object
            `~astropy.table.Table` storing halo catalog.

            After calling the `assign_phase_space` method,
            the `x`, `y`, `z`, `vx`, `vy`, and `vz`
            columns of the input ``table`` will be over-written
            with their host-centric values.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        """
        MonteCarloGalProf.mc_pos(self, table=table, seed=seed)
        if seed is not None:
            seed += 1
        self.mc_vel(table, seed=seed)

    def mc_vel(self, table, seed=None):
        r""" Method assigns a Monte Carlo realization of the Jeans velocity
        solution to the halos in the input ``table``.

        Parameters
        -----------
        table : Astropy Table
            `astropy.table.Table` object storing the halo catalog.
            Calling the `mc_vel` method will over-write the existing values of
            the ``vx``, ``vy`` and ``vz`` columns.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        """
        vx, vy, vz = MonteCarloGalProf.mc_vel(self, table, seed=seed, overwrite_table_velocities=False, return_velocities=True)
        if vx is None: 
            return 
        # scale velocity by eta_vs (satellite velocity bias) 
        vx *= self.param_dict['eta_vb.satellites']
        vy *= self.param_dict['eta_vb.satellites']
        vz *= self.param_dict['eta_vb.satellites']
    
        # add velcotiy to table.
        table['vx'][:] += vx
        table['vy'][:] += vy
        table['vz'][:] += vz

    def calculate_conc_gal_bias(self, seed=None, **kwargs):
        r""" Calculate the ratio of the galaxy concentration to the halo concentration,
        :math:`c_{\rm gal}/c_{\rm halo}`.
        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing the mass-like variable, e.g., ``halo_mvir``.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.
        table : object, optional
            `~astropy.table.Table` storing the halo catalog.
            If your NFW model is based on the virial definition,
            then ``halo_mvir`` must appear in the input table,
            and likewise for other halo mass definitions.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.
        Returns
        -------
        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`c_{\rm gal}/c_{\rm halo}`.
        """
        if 'table' in list(kwargs.keys()):
            table = kwargs['table']
            mass = table[self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop']).astype('f4')
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``assign_conc_gal_bias`` function of the ``BiasedNFWPhaseSpace`` class.\n")
            raise KeyError(msg)

        result = np.zeros_like(mass) + self.param_dict['conc_gal_bias.satellites']

        if 'table' in list(kwargs.keys()):
            table['conc_gal_bias'][:] = result
            halo_conc = table['conc_NFWmodel']
            gal_conc = self._clipped_galaxy_concentration(halo_conc, result)
            table['conc_galaxy'][:] = gal_conc
        else:
            return result
