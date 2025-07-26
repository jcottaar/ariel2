'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module defines and solves the Bayesian model used as the main model for my ARIEL work.
Sections:
- Model definition
- Model solving
- Support functions and classes
- Model visualization
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import ariel_support as ars # support and loading functions
import copy
import gp # my own GP library


'''
Model definition
'''
def baseline_model(include_later_optimization=False):
    # include_later_optizimation=False: as my best submission before the competition closed
    # include_later_optizimation=True: include later learnings
    modeli = PredictionModel()
    if not include_later_optimization:
        model = ars.SigmaFudger()
        model.model = ars.MeanBiasFitter()
        model.model.model = modeli
        return model
    else:
        return modeli
        
class ModelOptions(ars.LockedClass):
    # Configures define_prior and fit_gp

    use_training_labels = False # if True we make use of the ground truth; useful when we want to investigate the other prior elements or for visualization

    # Configuration of the transit depth prior
    update_transit_variation_sigma = True # whether to update the magnitude of the transit depth variation; this effectively makes the fit less aggressive if the transit depth seems flat
    min_transit_scaling_factor = 0.2 # minimum value for the magnitude scaling above; necessary because maximum likelihood estimation tends to underestimate small values
    n_pca = 1 # number of PCA components to use
    retrain_pca = True  # whether to retrain the PCA shapes (rather than use PCA on the training labels)
    transit_prior_info=0 # PCA shapes and magnitudes; loaded in constructor, overwritten if retrain_pca = True

    # Configuration of the drift prior
    do_not_set_drift_hyperparameters = False # used for tuning
    hfactor = 1 # determines the resolution of the KISS-GP grid for the 2D drift (spectral drift); higher hfactor means faster calculation and lower accuracy

    # Configuration of the solver
    n_samples_sigma_est = 100 # how many samples to take to estimate the covariance matrix (and hence the sigma prediction)
    n_iter = 7 # number of iterations for the non-linear solver
    update_rate = 1 # update rate 
    max_log_update_hyperparameters = 1 # maximum update of log(hyperparameters) in each iteration
    
    def __post_init__(self):
        try:
            self.transit_prior_info = ars.pickle_load(ars.file_loc() + 'transit_depth_gp_with_pca.pickle')
        except:
            pass # this comes up when determining the transit window function (code not shared on Kaggle)
        super().__post_init__()
        
        
def define_prior(obs, model_options, data):
    # This is the core function that defines the prior for the Bayesian Inference.
    #
    # Inputs:
    # obs: An Observable (as defined later in this module) containing the measured data
    # model_options: A ModelOptions configuring the model
    # data: the data as returned by ars.DataLoader
    #
    # Returns an initialized model as defined in gp.py, as well as un unitialized copy (used for some special flows)
    #
    # For an overview, see https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/543853
    
    # Overall model: (star_spectrum * drift * transit) + noise
    # We now define each of these one by one, then combine them

    
    ### Star spectrum - unregularized value per wavelength
    stellar_spectrum_model = gp.ParameterScaler()
    stellar_spectrum_model.scaler = 1e5 # Typical parameter scale, helps with numerics
    stellar_spectrum_model.model = gp.Uncorrelated() # The most basic class, defining uncorrelated Gaussian prior
    stellar_spectrum_model.model.features = ['wavelength']
    stellar_spectrum_model.model.sigma = 1e20 # 'infinite'


    
    
    ### Drift - we define the model separately for AIRS and for FGS, then combine them at the end
    
    ## AIRS drift: average (depends only on time) and spectral (depends on time and wavelength); we'll define them in 'm' and then create the final class at the end
    m = dict()
    # Define average AIRS drift - 1D Gaussian Processs over time
    if not model_options.do_not_set_drift_hyperparameters:
        drift_hyper = ars.pickle_load(ars.file_loc() + 'drift_hyperparameters.pickle')
    m['average'] = gp.SquaredExponentialKernelMulti() # our standard GP class, combining squared-exponential kernels for multiple length scales
    m['average'].features = ['time']
    m['average'].lengths = ars.pickle_load(ars.file_loc() + 'drift_hyperparameters_old.pickle')['average_lengths'] # lengths for the kernels are in this file for legacy reasons
    if not model_options.do_not_set_drift_hyperparameters:
        m['average'].sigmas = drift_hyper['average_sigmas'] # load tuned values
    else:
        m['average'].sigmas = init_drift_sigmas(m['average'].lengths) # only used during tuning, not included in Kaggle shared code
    m['average'].require_mean_of_non_noise_zero = True # we don't want the drift to catch offsets
    # Define spectral AIRS drift - 2D Gaussian Processs over time and wavelength
    m['spectral'] = gp.Sparse2D() # this class does the KISS-GP to speed things up
    m['spectral'].features = ['time', 'wavelength']        
    m['spectral'].model = gp.SquaredExponentialKernelMulti()
    m['spectral'].model.features = ['x','y'] # dummy features introduced by gp.Sparse2D()
    m['spectral'].model.lengths = ars.pickle_load(ars.file_loc() + 'drift_hyperparameters_old.pickle')['spectral_lengths']
    if not model_options.do_not_set_drift_hyperparameters:
        m['spectral'].model.sigmas = drift_hyper['spectral_sigmas']
        m['spectral'].model.sigmas[-1] = m['spectral'].model.sigmas[-1]*1e-2
    else:
        m['spectral'].model.sigmas = init_drift_sigmas(m['spectral'].model.lengths)
    m['spectral'].model.require_mean_of_non_noise_zero = True
    m['spectral'].h = np.array([0.4, 0.05])*model_options.hfactor # grid resolution for KISS-GP
    # Combine average and spectral drift
    drift_model_AIRS = gp.ParameterScaler()
    drift_model_AIRS.scaler = 1e-3 # Typical parameter scale, helps with numerics
    drift_model_AIRS.model = gp.CompoundNamed() # This class combines multiple models
    drift_model_AIRS.model.m = m # Insert the models defined above
    drift_model_AIRS.model.offset = 1. # Add 1 (drift is 1+average+spectral)
    drift_model_AIRS.model.mode = 'sum' # Indicates that we add (rather than multiply) the models

    ## FGS drift: average only, otherwise similar to above
    # Define average FGS drift - 1D Gaussian Processs over time
    m = dict()
    if not model_options.do_not_set_drift_hyperparameters:
        drift_hyper = ars.pickle_load(ars.file_loc() + 'drift_hyperparameters_fgs.pickle')
    m['average'] = gp.SquaredExponentialKernelMulti()
    m['average'].features = ['time']
    m['average'].lengths = ars.pickle_load(ars.file_loc() + 'drift_hyperparameters_fgs_old.pickle')['average_lengths']
    if not model_options.do_not_set_drift_hyperparameters:
        m['average'].sigmas = drift_hyper['average_sigmas']
    else:
        m['average'].sigmas = init_drift_sigmas(m['average'].lengths)
    m['average'].require_mean_of_non_noise_zero = True
    # Combine; not really needed since it's just 1 model, but we do it for consistency with AIRS model
    drift_model_FGS = gp.ParameterScaler()
    drift_model_FGS.scaler = 1e-3
    drift_model_FGS.model = gp.CompoundNamed()
    drift_model_FGS.model.m = m
    drift_model_FGS.model.offset = 1.
    drift_model_FGS.model.mode = 'sum'

    ## Create the final drift model by combining the AIRS and FGS models above
    drift_model = ModelSplitSensors() # This class handles the splitting of AIRS and FGS, applying the AIRS model to AIRS data and FGS model to FGS data
    m = dict()
    m['FGS'] = drift_model_FGS
    m['AIRS'] = drift_model_AIRS   
    drift_model.m = m
    
    
    
    ### Transit - This is the product of the transit depth and the transit window. We'll define those separately and then combine them.

    ## Transit depth model. This one's structure is a bit nested:
    ## - Average: offset
    ## - Variation: sum of...
    ##    - Non PCA part, split between FGS and AIRS as...
    ##       - FGS: offset (regularized)
    ##       - AIRS: 1D Gaussian Process over wavelength
    ##    - PCA part, consisting of fixed basis functions (the PCA shapes determind earlier)
    # Define average model
    m = dict() # this is where we will combine all the transit depth components
    m['average'] = gp.ParameterScaler()
    m['average'].scaler = 0.002 # helps with numerics
    m['average'].model = gp.Uncorrelated()
    m['average'].model.features = [] # doesn't depend on features -> single offset value
    m['average'].model.sigma = 0.01 # kind of infinite
    # Define the non-PCA part of the variation
    model_AIRS = gp.ParameterScaler()
    model_AIRS.scaler = np.sqrt(np.sum(model_options.transit_prior_info['sigmas_per_npca'][model_options.n_pca]**2))
    model_AIRS.model = gp.SquaredExponentialKernelMulti()
    model_AIRS.model.features = ['wavelength']
    model_AIRS.model.sigmas = model_options.transit_prior_info['sigmas_per_npca'][model_options.n_pca]
    model_AIRS.model.lengths = model_options.transit_prior_info['lengths']   
    model_FGS = gp.ParameterScaler()
    model_FGS.scaler = 1e-4
    model_FGS.model = gp.Uncorrelated()
    model_FGS.model.features = ['wavelength']
    model_FGS.model.sigma = model_options.transit_prior_info['fgs_sigmas'][model_options.n_pca]
    model_variation_non_pca = ModelSplitSensors()
    mm = dict()
    mm['FGS'] = model_FGS
    mm['AIRS'] = model_AIRS
    model_variation_non_pca.m = mm
    # Define the PCA part of the variation
    if model_options.n_pca>0:
        model_variation_pca = gp.ParameterScaler()
        model_variation_pca.scaler = 1e-4
        model_variation_pca.model = gp.FixedBasis()
        model_variation_pca.model.basis_functions = model_options.transit_prior_info['U'][:,:model_options.n_pca]
        model_variation_pca.model.regularization_variance = model_options.transit_prior_info['variances'][:model_options.n_pca]
        model_variation_pca.model.features = ['wavelength']     
    # Create the variation model
    m['variation'] = gp.CompoundNamed()
    mm = dict()
    mm['non_pca'] = model_variation_non_pca
    if model_options.n_pca>0:
        mm['pca'] = model_variation_pca
    m['variation'].m=mm
    m['variation'].update_scaling = model_options.update_transit_variation_sigma # Determine if we tune the magnitude of the variation, essentially scaling all sigma values within
    # Create the final transit depth model
    transit_depth_model = gp.CompoundNamed()
    transit_depth_model.m = m
    # And undo all our work above if we're using the training labels - in that case the prior is just a fixed value (delta distribution)
    if model_options.use_training_labels:
        # Use fixed transit depth
        transit_depth_model = gp.FixedValue()
        transit_depth_model.offset = np.nan*obs.df['time'].to_numpy()
        for i in range(len(data['labels'])):
            transit_depth_model.offset[np.isclose(obs.df['wavelength'], data['wavelengths_report'][i])] = -data['labels'][i]
        assert(not np.any(np.isnan(transit_depth_model.offset)))

    ## Transit window - a fixed pretrained function, where we do fit the ingress/egress times and width
    transit_window_model = TransitWindowModel() # handles the above
    transit_window_model.features = ['time']

    ## Combine the models above to create the full transit model
    m = dict()
    m['transit_depth'] = transit_depth_model        
    m['transit_window'] = transit_window_model
    transit_model = gp.CompoundNamed()
    transit_model.m = m
    transit_model.offset = 1.
    transit_model.mode = 'product' # multiply transit depth and transit window





    ### Noise model - uncorrelated Gaussian per point, with sigma values from preprocessing
    noise_model = gp.ParameterScaler()
    noise_model.scaler = 1. # noise must have design matrix 1 at present -> can't scale
    noise_model.model = NoiseModel() # this model also handles the different time binnings
    noise_model.model.features = ['time', 'wavelength']
    noise_model.model.features_hyper = 1 # indicates that hyperparameters (i.e. noise magnitudes) depend on wavelength
    noise_model.model.check_for_uniqueness = False # speeds things up
    noise_model.model.use_as_noise_matrix = True # indicates that this behaves as noise (solver needs this information)
    noise_model.model.sigma_varying = np.concatenate((data['FGS']['noise_est'], data['AIRS']['noise_est'])) # set sigma values
    



    
    ### Combine everything
    # Multiple star spectrum, drift, and transit
    m = dict()
    m['spectrum'] = stellar_spectrum_model
    m['drift'] = drift_model
    m['transit'] = transit_model
    signal_model = gp.CompoundNamed()
    signal_model.m=m
    signal_model.mode = 'product'
    signal_model.expected_observation_scale = 1e5 # for debugging, doesn't normally matter
    # Add noise to the above
    m = dict()
    m['signal']= signal_model
    m['noise'] = noise_model    
    model = gp.CompoundNamed()
    model.m=m
    model.expected_observation_scale = 1e5 # for debugging, doesn't normally matter

    model_uninitialized = copy.deepcopy(model)
    model.initialize(obs) # inform the models about the observable, so they can for example figure out how many parameters they have and initialize them to zero

    ## Set some sensible default values, this helps the non-linear solver to converge faster
    # Star spectrum: use the average of the signal per wavelength
    model.m['signal'].m['spectrum'].model.parameters = \
        np.concatenate(( np.reshape(np.mean(data['FGS']['data'], axis=0), (-1,1)), np.reshape(np.mean(data['AIRS']['data'], axis=0), (-1,1)) ))
    # Transit window: use the ingress and egress times estimates during preprocessing 
    model.m['signal'].m['transit'].m['transit_window'].base_values = np.array([data['t_ingress'], data['t_egress']   , 0])
    model.m['signal'].m['transit'].m['transit_window'].parameters = np.reshape(model.m['signal'].m['transit'].m['transit_window'].base_values, (-1,1))

    # Model sanity check
    model.check_constraints(debugging_mode_offset=1)

    return model, model_uninitialized



'''
Model solving
'''
def fit_gp(data, plot_final=False, model_options=ModelOptions(), stop_before_solve = False):
    # Performs the Bayesian Inference. 
    # Inputs:
    # -data: measurement data as obtained from ars.DataLoader()
    # -plot_final: whether to make some plots of the posterior
    # -model_options: configuration
    # -stop_before_solve: outputs the prior without solving it; used for some special flows
    # Returns the mean of and samples from the posterior
    
    data = copy.deepcopy(data)

    # Get our data into the right form for gp.py
    obs = Observable()
    obs.import_from_loaded_data(data)

    # Define the prior
    prior_model, model_uninitialized = define_prior(obs, model_options, data)    
    if stop_before_solve:
        # For external analysis
        return prior_model, model_uninitialized, obs

    # Call the solver in gp.py
    def adapt_func(model):
        # This function, provided to the solver, applies the minimum magnitude of the transit depth variation hyperparameter. The solver will run it after initial tuning.
        if not model_options.use_training_labels:
            model.m['signal'].m['transit'].m['transit_depth'].m['variation'].scaling_factor = \
                np.max((model_options.min_transit_scaling_factor, model.m['signal'].m['transit'].m['transit_depth'].m['variation'].scaling_factor))
        return model        
    posterior_mean, posterior_samples = gp.solve_gp_nonlinear(prior_model, obs, rng=np.random.default_rng(seed=data['planet_id']), \
        update_rate=model_options.update_rate, n_iter=model_options.n_iter, update_hyperparameters_from=0,\
        hyperparameter_method = 'gradient_descent', adapt_func = adapt_func, max_log_update_initial = model_options.max_log_update_hyperparameters,\
        n_samples = model_options.n_samples_sigma_est)

    # Plot if desired
    if plot_final:
        visualize_gp(obs, posterior_mean, posterior_samples, data, model_options)

    # Lots of sanity checks to catch misbehaving planets in the private test set
    if ars.sanity_checks_active:        
        obs_expected_noise_squared = copy.deepcopy(obs)
        obs_expected_noise_squared.labels = np.reshape(1/posterior_mean.m['noise'].get_prior_matrices(obs).prior_precision_matrix.diagonal(), (-1,1))
        obs_noise = copy.deepcopy(obs)
        obs_noise.labels =  posterior_mean.m['noise'].get_prediction(obs)

        # Total noise
        scaled_noise = obs_noise.labels / np.sqrt(obs_expected_noise_squared.labels)
        ars.sanity_check(np.max, np.abs(scaled_noise), 'scaled_noise_max', 21, [2, 8]) 
        ars.sanity_check(ars.rms, scaled_noise, 'scaled_noise_rms', 22, [0.95, 1.05]) 

        # FGS noise
        scaled_noise = obs_noise.export_matrix(False) / np.sqrt(obs_expected_noise_squared.export_matrix(False))
        ars.sanity_check(np.max, np.abs(scaled_noise), 'fgs_scaled_noise_max', 23, [1, 8]) 
        ars.sanity_check(ars.rms, scaled_noise, 'fgs_scaled_noise_rms', 24, [0.3, 1.7]) 

        # AIRS noise per column (summed over wavelength)
        scaled_noise = np.sum(obs_noise.export_matrix(True), axis=1) / np.sqrt(np.sum(obs_expected_noise_squared.export_matrix(True), axis=1))
        ars.sanity_check(np.max, np.abs(scaled_noise), 'airs_column_scaled_noise_max', 25, [1, 8]) 
        ars.sanity_check(ars.rms, scaled_noise, 'airs_column_scaled_noise_rms', 26, [0.7, 1.3]) 

        # AIRS noise per row (summed over time)
        scaled_noise = np.sum(obs_noise.export_matrix(True), axis=0) / np.sqrt(np.sum(obs_expected_noise_squared.export_matrix(True), axis=0))
        ars.sanity_check(np.max, np.abs(scaled_noise), 'airs_row_scaled_noise_max', 27, [0, 7]) 
        ars.sanity_check(ars.rms, scaled_noise, 'airs_row_scaled_noise_rms', 28, [0, 2]) 
        
        ars.sanity_check(np.min, posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['average'].parameters, 'drift_min', 15, [-5e-3, 0]) 
        ars.sanity_check(np.max, posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['average'].parameters, 'drift_max', 15, [0, 5e-3]) 
        ars.sanity_check(np.min, posterior_mean.m['signal'].m['drift'].m['FGS'].model.m['average'].parameters, 'drift_FGS_min', 14, [-2e-2, 0]) 
        ars.sanity_check(np.max, posterior_mean.m['signal'].m['drift'].m['FGS'].model.m['average'].parameters, 'drift_FGS_max', 14, [0, 2e-2]) 
        ars.sanity_check(np.min, posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['spectral'].model.parameters, 'spectral_drift_min', 16, [-2e-2, 0]) 
        ars.sanity_check(np.max, posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['spectral'].model.parameters, 'spectral_drift_max', 16, [0, 2e-2]) 
        ars.sanity_check(lambda x:x, posterior_mean.m['signal'].m['transit'].m['transit_window'].parameters[0]-data['t_ingress'], 'ingress_time_delta', 19, [-0.15,0.15])
        ars.sanity_check(lambda x:x, posterior_mean.m['signal'].m['transit'].m['transit_window'].parameters[1]-data['t_egress'], 'egress_time_delta', 19, [-0.15,0.15])
        ars.sanity_check(np.exp, posterior_mean.m['signal'].m['transit'].m['transit_window'].parameters[2], 'ingress_width', 18, [0,3])    

    # Compile results
    results = dict()
    results['obs'] = obs
    results['model_mean'] = posterior_mean # mean of the posterior
    results['model_samples'] = posterior_samples # samples from the posterior (amount determined by model_options.n_samples_sigma_est)
    return results


def tune_model_options(model_options, data):
    # Redoes parts of model tuning on the *test* set. Called by PredictionModel.infer_internal().
    if model_options.retrain_pca:
        # Redo the PCA on the test set

        # Define an adapted model without PCA
        model_temp = PredictionModel()
        model_temp.model_options = copy.deepcopy(model_options)
        model_temp.skip_tuning = True
        model_temp.model_options.retrain_pca = False
        model_temp.model_options.n_pca = 0
        model_temp.model_options.n_iter = model_temp.model_options.n_iter//2 # less iterations, we don't hwave all day
        model_temp.model_options.max_log_update_hyperparameters = 2*model_temp.model_options.max_log_update_hyperparameters # faster convergence of hyperparameters
        model_temp.model_options.n_samples_sigma_est = 2 # we don't need samples at all, but other code gets confused if there less than 2

        # Run the model        
        transit_prior_info = copy.deepcopy(model_temp.model_options.transit_prior_info)
        model_temp.train(data)
        pred,sigma,cov = model_temp.infer(data)

        # Redo the PCA. We use SVD (=uncentered PCA)
        data_mean_removed = pred.T-np.mean(pred.T,axis=0) # Removing the mean per planet, but not the mean per wavelength (so PCA is uncentered)
        U,S,Vh = np.linalg.svd(data_mean_removed)
        transit_prior_info['U'] = U # PCA shapes
        transit_prior_info['variances'] = S**2/data_mean_removed.shape[1] # PCA magnitudes, used for regularizatoin

        # Update model_options for the final fit
        model_options.transit_prior_info = transit_prior_info

    return model_options


'''
Support functions and classes
'''
class PredictionModel(ars.Model):
    # Wrapper around the functionality in the model to match the interface defined by ars.Model()

    model_options = None # ModelOptions as defined earlier, configuring the model
    skip_tuning = False # used internally

    def __init__(self):
        super().__init__()
        self.model_options = ModelOptions()

    def train_internal(self, train_data):
        pass # nothing to do 

    def infer_internal(self, test_data):
        if not self.skip_tuning:
            # Adjust model options if needed, like retraining PCA shapes
            self.model_options=tune_model_options(self.model_options, test_data)     

        # Do the actual inference, which will end up calling infer_internal_single below per planet
        return super().infer_internal(test_data)

    def infer_internal_single(self, test_data):
        # Construct and fithe GPs
        results = fit_gp(test_data, model_options=self.model_options)       

        # Find where in the output we can find the desired transit depth values
        inds = ars.ismembertol(test_data['wavelengths_report'], results['obs'].df['wavelength'])
        inds = np.delete(inds, inds==-1)
        
        # Evaluate the transit depth part of the posterior only, using the **mean** of the posterior
        pred_labels = results['model_mean'].m['signal'].m['transit'].m['transit_depth'].get_prediction(results['obs'])

        # Grab the correct part of it -> we have our prediction
        pred = -pred_labels[inds,0]

        # Same as above, but now using **samples** to estimate the covariance matrix and sigma predictions
        sample_labels = results['model_samples'].m['signal'].m['transit'].m['transit_depth'].get_prediction(results['obs'])
        sample_labels = sample_labels[inds,:]-pred_labels[inds,0][:,np.newaxis]
        cov = (sample_labels@sample_labels.T)/sample_labels.shape[1]
        sigma = np.std(sample_labels, axis=1)

        # Sanity checks
        ars.sanity_check(np.mean, pred, 'pred_mean', 10, [0, 0.03])
        ars.sanity_check(np.std, pred, 'pred_std', 6, [0, 0.005])
        if self.model_options.n_samples_sigma_est>50:
            ars.sanity_check(np.min, sigma, 'sigma_min', 8, [7e-6, 1e-4])
            ars.sanity_check(np.max, sigma, 'sigma_max', 9, [7e-6, 0.0007])

        return pred,sigma,cov


class Observable(gp.Observable):
    # Casts the measued data in the form expected by the gp library
    # Specifically, the df propery will be a pandas dataframe, with one row per measurement, and columns:
    # -time: timestamp
    # -wavelength: wavelength
    # -points_ber_bin: number of frames that were binned for this measurement - needed for noise prior
    # -is_AIRS: True for AIRS, False for FGS
    # Finally, the labels property will contain the actualy measured values per pixel.
    AIRS_shape = 0 # cache the shape of the AIRS matrix to be able to export it later
    FGS_shape = 0 # cache the shape of the FGS matrix to be able to export it later
    
    def import_from_loaded_data(self, data):
        # Load from 'data' as output by ars.DataLoader
        def get_df(do_AIRS):
            # Get dataframe for one sensor
            if do_AIRS:
                field = 'AIRS'
            else:
                field = 'FGS'

            # Get times and wavelengths into the proper column shape
            times = data[field]['times'];
            times = np.tile(np.reshape(times,(-1,1)), (1,len(data[field]['wavelengths'])))
            points_per_bin = np.tile(np.reshape(data[field]['points_per_bin'],(-1,1)), (1,len(data[field]['wavelengths'])))
            wavelengths = data[field]['wavelengths'];
            wavelengths = np.tile(wavelengths, (len(data[field]['times']), 1))

            # Make dataframe
            df = pd.DataFrame()
            df['time'] = np.reshape(times, (-1))
            df['points_per_bin'] = np.reshape(points_per_bin, (-1))
            df['wavelength'] = np.reshape(wavelengths, (-1))
            df['is_AIRS'] = np.full(len(df), do_AIRS)
            return df

        # Combine the two sensors
        self.df = pd.concat((get_df(False), get_df(True)))
        
        # Add labels
        self.labels = np.zeros((self.df.shape[0],1))
        self.import_matrix(data['FGS']['data'], False)
        self.import_matrix(data['AIRS']['data'], True)            

        # Sanity check
        self.check_constraints(debugging_mode_offset=1)

    def import_matrix(self,mat,is_AIRS):
        # Make the matrix of measurements a column
        self.labels[self.df['is_AIRS']==is_AIRS,:] = np.reshape(mat,(-1,1))
        if is_AIRS:
            self.AIRS_shape = mat.shape
        else:
            self.FGS_shape = mat.shape
        if gp.debugging_mode >= 2:
            assert np.all(mat == self.export_matrix(is_AIRS))

    def export_matrix(self, is_AIRS, instance=0):
        # Get the relevant sensor and return it as a matrix
        if is_AIRS:
            return np.reshape(self.labels[self.df['is_AIRS'],instance], self.AIRS_shape)
        else:
            return np.reshape(self.labels[np.logical_not(self.df['is_AIRS']),instance], self.FGS_shape)

class NoiseModel(gp.UncorrelatedVaryingSigma):
    # gp.UncorrelatedVaryingSigma handles different noise per wavelength; this class adds noise scaling with the number of frames combined per time bin
    def get_prior_distribution_internal(self, obs):
        prior_matrices = super().get_prior_distribution_internal(obs)        
        prior_matrices.prior_precision_matrix = gp.sparse_matrix ( sp.sparse.diags(obs.df['points_per_bin'].to_numpy()) @ prior_matrices.prior_precision_matrix ) 
        return prior_matrices

class TransitWindowModel(gp.FeatureSelector):
    # Manages the transit window. It uses a pretrained fixed ingress/egress function, and shifts and widens it according to its 3 parameters: ingress time, egress time, and width.

    base_values = None # Mean of the prior; will be set to initial values. Not particularly important.
    std_values = 1/np.array([0.05, 0.05, 0.01]) # I think I may have made a 1/x error here, but these are effectively infinite anyway
    
    def get_number_of_parameters_from_mat_internal(self, mat): 
        return 3

    def get_observable_relationship_from_mat_internal(self,mat):
        # Brute force: finite differences per parameter
        assert self.number_of_instances==1
        prior_matrices = gp.PriorMatrices()
        prior_matrices.number_of_observations = mat.shape[0]      
        params_old = copy.copy(self.parameters)
        pred_base = self.get_prediction_from_mat_internal(mat)
        step_size = 1e-6
        res = np.zeros((mat.shape[0], self.number_of_parameters))
        for i in range(self.number_of_parameters):
            self.parameters = copy.copy(params_old)
            self.parameters[i,0] = self.parameters[i,0]+step_size
            pred_new = self.get_prediction_from_mat_internal(mat)
            res[:,i] = np.reshape((pred_new-pred_base)/step_size, -1)
        self.parameters = params_old
        prior_matrices.design_matrix = gp.sparse_matrix(res)
        prior_matrices.observable_offset = pred_base - prior_matrices.design_matrix@self.get_parameters()   
        prior_matrices.observable_offset = np.reshape( prior_matrices.observable_offset,-1 )
        return prior_matrices

    def get_prior_distribution_from_mat_internal(self,mat):
        # Prior: maen is base_values, standard deviation is std_values
        prior_matrices = gp.PriorMatrices()        
        prior_matrices.number_of_parameters = self.get_number_of_parameters_from_mat_internal(mat)
        prior_matrices.prior_mean = self.base_values
        prior_matrices.prior_precision_matrix = gp.sparse_matrix ( sp.sparse.diags(1/self.std_values**2))
        return prior_matrices

    def get_prediction_from_mat_internal(self, mat):        
        # With parameters given, predict on the time values given in mat
        
        def transit_window_function(t):
            # The fixed function to use
            transit_splines3 = ars.pickle_load(ars.file_loc()+'ingress_egress_window3.pickle') # pretrained
            r = 0*t
            todo = (t>=0)
            if np.any(todo):
                r[todo] = transit_splines3(t[todo])
            todo = (t<0)
            if np.any(todo):
                r[todo] = 1-transit_splines3(-t[todo])
            r[r<0] = 0
            r[r>1] = 1
            return r
            
        res = np.zeros((mat.shape[0], self.number_of_instances)) + np.nan
        for i in range(self.number_of_instances): # self.number_of_instances is the number of samples, usually it will be 1 when calling this
            this_res = res[:,i]
            vals = mat[:,0]
            t1 = self.parameters[0,i] # ingress time
            t2 = self.parameters[1,i] # egress time
            w1 = np.exp(self.parameters[2,i]) # ingress width
            w2 = np.exp(self.parameters[2,i]) # egress width = ingress width
            this_res[vals<(t1+t2)/2] = transit_window_function((vals[vals<(t1+t2)/2]-t1)/w1)
            this_res[vals>=(t1+t2)/2] = 1-transit_window_function((vals[vals>=(t1+t2)/2]-t2)/w2)
            this_res[this_res==0] = 1e-30 # gp.py doesn't like zeros sometimes
            res[:,i] = this_res
        return res

    def is_linear(self):
        return(False) # indicates our predictions are not a linear function of our parameters


class ModelSplitSensors(gp.CompoundNamed):
    # Applies m['AIRS'] to AIRS part of data, m['FGS'] to FGS part of data. This involves lots of code, but that's all it does.
    def check_constraints_internal(self):
        assert len(self.models)==2
        # check keys
        self.m['AIRS']
        self.m['FGS']
        assert isinstance(self.offset, float)
        assert isinstance(self.models, list)
        assert len(self.models)>0
        assert self.number_of_parameters == np.sum([x.number_of_parameters for x in self.models])
        if not self.update_scaling:
            assert self.number_of_hyperparameters == np.sum([x.number_of_hyperparameters for x in self.models])
        else:
            assert self.number_of_hyperparameters == 1
        assert self.mode == 'sum' or self.mode == 'product'        
        assert np.all(self.number_of_instances == np.array([x.number_of_instances for x in self.models]))
        for m in self.models:
            assert isinstance(m, gp.Model)
            if gp.debugging_mode>=2:
                m.check_constraints()

    def split_obs(self, obs):
        obs_AIRS = copy.deepcopy(obs)
        obs_AIRS.select_observations(obs.df['is_AIRS'])
        obs_FGS = copy.deepcopy(obs)
        obs_FGS.select_observations(np.logical_not(obs.df['is_AIRS']))
        return obs_AIRS, obs_FGS

    def initialize_internal(self, obs, number_of_instances):
        obs_AIRS, obs_FGS = self.split_obs(obs)
        param = 0
        hparam = 0
        self.m['AIRS'].initialize(obs_AIRS, number_of_instances=number_of_instances)
        param = param+self.m['AIRS'].number_of_parameters
        hparam = hparam+self.m['AIRS'].number_of_hyperparameters
        self.m['FGS'].initialize(obs_FGS, number_of_instances=number_of_instances)
        param = param+self.m['FGS'].number_of_parameters
        hparam = hparam+self.m['FGS'].number_of_hyperparameters
        self.number_of_parameters = param
        self.number_of_hyperparameters = hparam

    def get_observation_relationship_internal(self,obs):
        obs_AIRS, obs_FGS = self.split_obs(obs)
        pm_AIRS = self.m['AIRS'].get_prior_matrices(obs_AIRS)
        pm_FGS = self.m['FGS'].get_prior_matrices(obs_FGS)
        prior_matrices = gp.PriorMatrices()
        
        prior_matrices.number_of_observations = pm_AIRS.number_of_observations + pm_FGS.number_of_observations

        prior_matrices.observable_offset = np.zeros((self.number_of_observations))
        prior_matrices.observable_offset[obs.df['is_AIRS']] = pm_AIRS.observable_offset
        prior_matrices.observable_offset[np.logical_not(obs.df['is_AIRS'])] = pm_FGS.observable_offset
        prior_matrices.observable_offset = prior_matrices.observable_offset + self.offset
        prior_matrices.design_matrix = sp.sparse.block_diag([pm_FGS.design_matrix, pm_AIRS.design_matrix], format=gp.sparse_matrix_str)
        assert np.all(np.diff(obs.df['is_AIRS'])>=0)
        

        return prior_matrices

    def get_prior_distribution_internal(self,obs):
        obs_AIRS, obs_FGS = self.split_obs(obs)
        assert(self.model_names == ['FGS', 'AIRS'])
        pm = [self.models[0].get_prior_matrices(obs_FGS), self.models[1].get_prior_matrices(obs_AIRS)]
        prior_matrices = gp.PriorMatrices()
        
        prior_matrices.number_of_observations = pm[0].number_of_observations + pm[1].number_of_observations
      
        prior_matrices.number_of_parameters = sum([p.number_of_parameters for p in pm])
        
        prior_matrices.prior_mean = np.concatenate([p.prior_mean for p in pm])
        prior_matrices.prior_precision_matrix = sp.sparse.block_diag([p.prior_precision_matrix for p in pm], format=gp.sparse_matrix_str)
        prior_matrices.noise_parameter_indices = np.zeros((0))
        is_noise = [len(p.noise_parameter_indices)>0 for p in pm]
        if sum(is_noise)==1:
            ind = np.argmax(is_noise)
            nparams_lower = sum([p.number_of_parameters for p in pm[0:ind]])
            prior_matrices.noise_parameter_indices = pm[ind].noise_parameter_indices + nparams_lower
        else:
            assert sum(is_noise)==0

        return prior_matrices

    def get_prediction_internal(self, obs):
        obs_AIRS, obs_FGS = self.split_obs(obs)
        res = np.zeros((self.number_of_observations, self.number_of_instances))
        res[obs.df['is_AIRS'],:] = self.m['AIRS'].get_prediction(obs_AIRS)
        res[np.logical_not(obs.df['is_AIRS']),:] = self.m['FGS'].get_prediction(obs_FGS)
        return res

def init_drift_sigmas(lengths):
    # defaults used when tuning hyperparameters; this function is used in define_prior
    sigmas = [1e-4 for x in lengths]
    sigmas.append(1e-7)
    return np.array(sigmas)



'''
Model visualization
'''
def visualize_gp(obs, posterior_mean, posterior_samples, data, model_options, simple=False):
    # Visualizes the model fit. See fit_gp for usage example.
    
    def model_to_mean_over_wavelengths(model, instance=0):
        obs_temp = copy.deepcopy(obs)
        obs_temp.labels = model.get_prediction(obs)
        return apply_func_by_wavelength(np.mean, obs_temp, instance=instance)
        

    def apply_func_by_wavelength(func, obs, instance=0):
        wl_unique,inds = np.unique(obs.df['wavelength'], return_inverse=True)
        result = []
        for i in range(len(wl_unique)):
            this_labels = obs.labels[obs.df['wavelength'] == wl_unique[i], instance]
            result.append(func(this_labels))
        return wl_unique, np.array(result)

    includes_FGS = np.any(np.logical_not(obs.df['is_AIRS']))
    includes_AIRS = np.any(obs.df['is_AIRS'])
    
    obs_signal = copy.deepcopy(obs)
    obs_signal.labels = posterior_mean.m['signal'].get_prediction(obs)
    obs_noise = copy.deepcopy(obs)
    obs_noise.labels = obs.labels - obs_signal.labels
    noise_AIRS = obs_noise.export_matrix(True)
    noise_FGS = obs_noise.export_matrix(False)

    obs_signal_sample = copy.deepcopy(obs)
    obs_signal_sample.labels = posterior_samples.m['signal'].get_prediction(obs)
    obs_noise_sample = copy.deepcopy(obs)
    obs_noise_sample.labels = obs_noise_sample.labels - obs_signal_sample.labels[:,0:1]

    model_no_transit = copy.deepcopy(posterior_mean)
    assert(model_no_transit.m['signal'].model_names[2] == 'transit')
    model_no_transit.m['signal'].models[2] = gp.FixedValue()
    model_no_transit.m['signal'].models[2].initialize(obs)
    model_no_transit.m['signal'].models[2].offset = 1.
    obs_no_transit = copy.deepcopy(obs)
    obs_no_transit.labels = model_no_transit.m['signal'].get_prediction(obs)

    # Means over dimensions    
    _,ax = plt.subplots(3,3,figsize=(18,18))
    for i in range(3):
        if i==0:
            plt.sca(ax[i,0])
            # plt.plot(obs.wavelengths, np.mean(total, axis=0))
            # plt.plot(obs.wavelengths, np.mean(signal, axis=0))
            wl_unique, result = apply_func_by_wavelength(np.mean, obs)
            plt.plot(wl_unique, result)
            plt.grid(True)
            plt.ylabel('Mean over time')
            plt.xlabel('Wavelength [um]')
            plt.legend(['Total', 'Signal'])
    
        for do_AIRS in [False, True]:
            if do_AIRS:
                if not includes_AIRS:
                    continue
                plt.sca(ax[i,1])
                plt.title('AIRS')
            else:
                if not includes_FGS:
                    continue
                plt.sca(ax[i,2])
                plt.title('FGS')
            times = np.unique(obs.df['time'][obs.df['is_AIRS']==do_AIRS])
            plt.plot(times, np.mean(obs.export_matrix(do_AIRS), axis=1))
            plt.plot(times, np.mean(obs_signal.export_matrix(do_AIRS), axis=1))
            plt.plot(times, np.mean(obs_no_transit.export_matrix(do_AIRS), axis=1))        
            plt.grid(True)
            plt.ylabel('Mean over wavelengths')
            plt.xlabel('Time [h]')
            plt.legend(['Total', 'Signal', 'Signal (without transit)'])
            if i==1:
                plt.xlim([data['t_ingress']-0.4, data['t_ingress']+0.4])
            if i==2:
                plt.xlim([data['t_egress']-0.4, data['t_egress']+0.4])


    if simple:
        return
    
    # transit
    _,ax = plt.subplots(1,2,figsize=(12,6))
    plt.sca(ax[0])
    for i in range(5):
        wl_unique, result = model_to_mean_over_wavelengths(posterior_samples.m['signal'].m['transit'].m['transit_depth'], instance=i)
        if i==0:
            plt.plot(wl_unique, -result, color = 'blue', linewidth=0.5)
        else:
            plt.plot(wl_unique, -result, color = 'blue', linewidth=0.5, label='_nolegend_')
    wl_unique, result = model_to_mean_over_wavelengths(posterior_mean.m['signal'].m['transit'].m['transit_depth'])
    plt.plot(wl_unique, -result, color='black')
    vals_pred = -result
    plt.grid(True)
    plt.ylabel('Mean over time')
    plt.xlabel('Wavelength [um]')
    plt.legend(['Transit depth'])
    if 'labels' in data.keys():
        vals_corr = data['labels']
        plt.plot(data['wavelengths_report'],vals_corr, color=[0.5,0.5,0.5])
        plt.legend(['Transit depth (sample from posterior)', 'Transit depth (mean of posterior)', 'Training labels (correct answer)'])
        try:
            plt.title('RMS error = '+str(ars.rms(vals_corr-vals_pred)))
        except:
            pass


    plt.sca(ax[1])    
    plt.scatter(obs.df['time'], posterior_mean.m['signal'].m['transit'].m['transit_window'].get_prediction(obs))
    plt.grid(True)
    plt.ylabel('Mean over wavelengths')
    plt.xlabel('Time [h]')
    plt.legend(['Transit window'])

    # drift
    _,ax = plt.subplots(1,2,figsize=(12,6))
    plt.sca(ax[0])
    obs_AIRS, obs_FGS =  posterior_mean.m['signal'].m['drift'].split_obs(obs)
    plt.scatter(obs_AIRS.df['time'], posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['average'].get_prediction(obs_AIRS))
    plt.scatter(obs_FGS.df['time'], posterior_mean.m['signal'].m['drift'].m['FGS'].model.m['average'].get_prediction(obs_FGS))
    plt.grid(True)
    plt.ylabel('Mean over wavelengths')
    plt.xlabel('Time [h]')
    plt.legend(['AIRS', 'FGS'])
    plt.title('Non-spectral drift')  
    if 'spectral' in posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m:
        plt.sca(ax[1])
        obs_AIRS.labels = posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['spectral'].get_prediction(obs_AIRS) #+posterior_mean.m['signal'].m['drift'].m['AIRS'].model.m['average'].get_prediction(obs_AIRS)
        plt.imshow(obs_AIRS.export_matrix(True), interpolation='none', aspect='auto')
        plt.colorbar()
        plt.title('Spectral drift (AIRS)')

    noise_prior_sample = gp.sample_from_prior(posterior_mean.m['noise'], obs, n_samples=1)
    obs_noise_prior_sample = copy.deepcopy(obs)
    obs_noise_prior_sample.labels = noise_prior_sample.get_prediction(obs)

    obs_signal_sample = copy.deepcopy(obs)
    obs_signal_sample.labels = posterior_samples.m['signal'].get_prediction(obs)
    obs_signal_sample.labels = obs_signal_sample.labels[:,0:1]
    obs_noise = copy.deepcopy(obs)
    obs_noise.labels = obs.labels - obs_signal_sample.labels    
    
    #AIRS noise
    _,ax = plt.subplots(2,2,figsize=(18,18))
    noise_mat = obs_noise.export_matrix(True)
    cbl= []
    cbu = []
    for column in [0,1]:
        synth_string = ''
        if column == 1:
            synth_string = ' (synthetic)'
            noise_mat = obs_noise_sample.export_matrix(True)
        #noise_mat = obs.export_AIRS_matrix()
        #noise_mat = np.random.default_rng(seed=0).normal(size=noise_mat.shape)*np.std(noise_mat)
        plt.sca(ax[0,column])
        plt.imshow(noise_mat, aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title('Noise' + synth_string)
        vmin, vmax = plt.gci().get_clim()
        cbl.append(vmin)
        cbu.append(vmax)
        if column == 1:
            plt.clim(cbl[0], cbu[0])

        plt.sca(ax[1,column])
        noise_filtered = ars.gaussian_2D_filter_with_nans(noise_mat, [15,4])
        plt.imshow(noise_filtered, aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title('Noise (low pass filtered)' + synth_string)
        vmin, vmax = plt.gci().get_clim()
        cbl.append(vmin)
        cbu.append(vmax)
        if column == 1:
            plt.clim(cbl[1], cbu[1])

    _,ax = plt.subplots(1,2,figsize=(12,6))

    plt.sca(ax[0])
    wl_unique, result = apply_func_by_wavelength(np.sum, obs_noise)
    plt.scatter(wl_unique, result)
    plt.xlabel('Wavelength [um]')
    plt.ylabel('Sum of noise term')
    obs_temp = copy.deepcopy(obs_noise)
    obs_temp.labels = 1/np.reshape(posterior_mean.m['noise'].get_prior_matrices(obs_temp).prior_precision_matrix.diagonal(), (-1,1)) # covariance
    wl_unique, result = apply_func_by_wavelength(lambda x:np.sqrt(np.sum(x)), obs_temp)
    plt.plot(wl_unique, result, color='black')
    plt.plot(wl_unique, -result, color='black')
    plt.grid(True)

    plt.sca(ax[1])
    wl_unique, result1 = apply_func_by_wavelength(np.std, obs_noise)
    plt.plot(wl_unique, result1)
    wl_unique, result2 = apply_func_by_wavelength(np.std, obs_noise_prior_sample)
    plt.plot(wl_unique, result2)
    plt.legend(['Posterior', 'Prior'])
    plt.xlabel('Wavelength [um]')
    plt.ylabel('STD of noise term')
    plt.grid(True)

    return result1[0], result2[0]