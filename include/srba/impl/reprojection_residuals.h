/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#pragma once

using namespace std;

namespace srba {

/** reprojection_residuals */
template <class KF2KF_POSE_TYPE,class LM_TYPE,class OBS_TYPE,class RBA_OPTIONS>
double RbaEngine<KF2KF_POSE_TYPE,LM_TYPE,OBS_TYPE,RBA_OPTIONS>::reprojection_residuals(
	vector_residuals_t & residuals, // Out:
	const std::vector<TObsUsed> & observations // In:
	) const
{
	const size_t nObs = observations.size();
	if (residuals.size()!=nObs) residuals.resize(nObs);

	double total_sqr_err = 0;

    //std::ofstream f;                                                                                //RGO_DBG
    //f.open("/home/ruben/work/programs/srba-stereo-slam/experiments/residual_errors/residuals.txt"); //RGO_DBG

	for (size_t i=0;i<nObs;i++)
	{
		// Actually measured pixel coords: observations[i]->obs.px
		const TKeyFrameID  obs_frame_id = observations[i].k2f->obs.kf_id; // Observed from here.
		const TRelativeLandmarkPos *feat_rel_pos = observations[i].k2f->feat_rel_pos;

		ASSERTDEB_(feat_rel_pos!=NULL)

		const TKeyFrameID  base_id  = feat_rel_pos->id_frame_base;

		pose_t const * base_pose_wrt_observer=NULL;

		// This case can occur with feats with unknown rel.pos:
		if (base_id==obs_frame_id)
		{
			base_pose_wrt_observer = &aux_null_pose;
		}
		else
		{
			// num[SOURCE] |--> map[TARGET] = CPose3D of TARGET as seen from SOURCE
			const typename TRelativePosesForEachTarget::const_iterator itPoseMap_for_base_id = rba_state.spanning_tree.num.find(obs_frame_id);
			ASSERT_( itPoseMap_for_base_id != rba_state.spanning_tree.num.end() )

			const typename frameid2pose_map_t::const_iterator itRelPose = itPoseMap_for_base_id->second.find(base_id);
			ASSERT_( itRelPose != itPoseMap_for_base_id->second.end() )

			base_pose_wrt_observer = &itRelPose->second.pose;
		}

		// pose_robot2sensor(): pose wrt sensor = pose_wrt_robot (-) sensor_pose_on_the_robot
		typename options::internal::resulting_pose_t<typename RBA_OPTIONS::sensor_pose_on_robot_t,REL_POSE_DIMS>::pose_t base_pose_wrt_sensor(mrpt::poses::UNINITIALIZED_POSE);
		RBA_OPTIONS::sensor_pose_on_robot_t::pose_robot2sensor( *base_pose_wrt_observer, base_pose_wrt_sensor, this->parameters.sensor_pose );

		const array_obs_t & real_obs = observations[i].k2f->obs.obs_arr;
		residual_t &delta = residuals[i];

		// Generate observation and compare to real obs:
		sensor_model_t::observe_error(delta,real_obs, base_pose_wrt_sensor,feat_rel_pos->pos, this->parameters.sensor);

        //f << std::sqrt( delta[0]*delta[0] + delta[1]*delta[1] ) << " " << std::sqrt( delta[2]*delta[2] + delta[3]*delta[3] )  << std::endl; //RGO_DBG
        //f << std::sqrt( delta[0]*delta[0] + delta[1]*delta[1] ) << std::endl;
        //f << std::sqrt( delta[2]*delta[2] + delta[3]*delta[3] ) << std::endl; //RGO_DBG

		const double sum_2 = delta.squaredNorm();
		if (this->parameters.srba.use_robust_kernel)
		{
			const double nrm = std::max(1e-11,std::sqrt(sum_2));
			//const double w = std::sqrt(huber_kernel(nrm,parameters.srba.kernel_param))/nrm;
			const double w = std::sqrt(cauchy_loss_kernel(nrm))/nrm;
			delta *= w;
			total_sqr_err += (w*w)*sum_2;
		}
		else
		{
			// nothing else to do:
			total_sqr_err += sum_2;
		}
	} // end for i

    //f.close();  //RGO_DBG

	return total_sqr_err;
}

/** reprojection_residuals and estimation of Gamma-distribution parameters */
template <class KF2KF_POSE_TYPE,class LM_TYPE,class OBS_TYPE,class RBA_OPTIONS>
double RbaEngine<KF2KF_POSE_TYPE,LM_TYPE,OBS_TYPE,RBA_OPTIONS>::reprojection_residuals(
		vector_residuals_t & residuals, // Out:
		const std::vector<TObsUsed> & observations, // In:
		vector_weights_t & weights, // Out
		double & stdv               // Out
		)
{

	const size_t nObs = observations.size();
	if (residuals.size()!=nObs) residuals.resize(nObs);
	if (weights.size()!=nObs)   weights.resize(nObs);

	Eigen::VectorXf residual_aux(2*nObs), residual_sort(2*nObs), residual_(2*nObs);

	double total_sqr_err = 0;

	//std::ofstream f;                                                                                 //RGO_DBG
	//f.open("/home/ruben/work/programs/srba-stereo-slam/experiments/residual_errors/residuals.txt");  //RGO_DBG

	//std::ofstream f1;                                                                                //RGO_DBG
	//f1.open("/home/ruben/work/programs/srba-stereo-slam/experiments/residual_errors/weights.txt");   //RGO_DBG

	for (size_t i=0;i<nObs;i++)
	{
		// Actually measured pixel coords: observations[i]->obs.px
		const TKeyFrameID  obs_frame_id = observations[i].k2f->obs.kf_id; // Observed from here.
		const TRelativeLandmarkPos *feat_rel_pos = observations[i].k2f->feat_rel_pos;

		ASSERTDEB_(feat_rel_pos!=NULL)

		const TKeyFrameID  base_id  = feat_rel_pos->id_frame_base;

		pose_t const * base_pose_wrt_observer=NULL;

		// This case can occur with feats with unknown rel.pos:
		if (base_id==obs_frame_id)
		{
			base_pose_wrt_observer = &aux_null_pose;
		}
		else
		{
			// num[SOURCE] |--> map[TARGET] = CPose3D of TARGET as seen from SOURCE
			const typename TRelativePosesForEachTarget::const_iterator itPoseMap_for_base_id = rba_state.spanning_tree.num.find(obs_frame_id);
			ASSERT_( itPoseMap_for_base_id != rba_state.spanning_tree.num.end() )

					const typename frameid2pose_map_t::const_iterator itRelPose = itPoseMap_for_base_id->second.find(base_id);
			ASSERT_( itRelPose != itPoseMap_for_base_id->second.end() )

					base_pose_wrt_observer = &itRelPose->second.pose;
		}

		// pose_robot2sensor(): pose wrt sensor = pose_wrt_robot (-) sensor_pose_on_the_robot
		typename options::internal::resulting_pose_t<typename RBA_OPTIONS::sensor_pose_on_robot_t,REL_POSE_DIMS>::pose_t base_pose_wrt_sensor(mrpt::poses::UNINITIALIZED_POSE);
		RBA_OPTIONS::sensor_pose_on_robot_t::pose_robot2sensor( *base_pose_wrt_observer, base_pose_wrt_sensor, this->parameters.sensor_pose );

		const array_obs_t & real_obs = observations[i].k2f->obs.obs_arr;
		residual_t &delta = residuals[i];

		// Generate observation and compare to real obs:
		sensor_model_t::observe_error(delta,real_obs, base_pose_wrt_sensor,feat_rel_pos->pos, this->parameters.sensor);

		// Save residual vector to estimate Gamma parameters
		residual_(2*i)   = std::sqrt( delta[0]*delta[0] + delta[1]*delta[1] );
		residual_(2*i+1) = std::sqrt( delta[2]*delta[2] + delta[3]*delta[3] );
		//std::cout << std::endl << delta << "\t" << residuals[i] << "\t" << residuals.size() << std::endl << std::endl;                                    //RGO_DBG
		//f << std::sqrt( delta[0]*delta[0] + delta[1]*delta[1] ) << std::endl << std::sqrt( delta[2]*delta[2] + delta[3]*delta[3] )  << std::endl;         //RGO_DBG

	} // end for i

	// MAD estimation of the standard deviation
	residual_aux  = residual_;
	residual_sort = residual_;
	std::sort(residual_sort.derived().data(),residual_sort.derived().data()+residual_sort.size());
	double median = residual_sort(nObs);
	residual_aux << (residual_aux - Eigen::VectorXf::Constant(2*nObs,median)).cwiseAbs();
	std::sort(residual_aux.derived().data(),residual_aux.derived().data()+residual_aux.size());
	stdv = 1.4826f * residual_aux(nObs);
	double var  = stdv*stdv;

	// Robust mean estimation
	double res  = 0.f;
	int samples = 0;
	for(unsigned int i = 0; i < 2*nObs; i++)
	{
		if(residual_(i)<1.f*stdv)
		{
			samples++;
			res  += residual_(i);
		}
	}

	// Gamma parameters estimation
	double mean  = res / (double)(samples);
	double theta = var / std::max(1e-11,mean);
	double alpha = mean*mean / std::max(1e-11,var) - 1.f;  // Since we only use it to estimate weights, we estimate alpha-1

	//f1 << var << "\t" << mean << "\t" << theta << "\t" << alpha+1.f << endl;
	//std::cout << std::endl << "Gamma parameters: \t alpha = "  << alpha+1.f << " \t theta = " << theta << " \t mean = " << mean << " \t var = " << var << std::endl << std::endl;

	// Weights assignation - TODO: find some metric indicating whether the distribution is accurately fitted or not (probably KS-test)
	if( alpha+1.f > 0.f && theta > 0.f && theta < 2.f )
	{

		std::cout << std::endl << "Gamma parameters: \t alpha = "  << alpha+1.f << " \t theta = " << theta << " \t mean = " << mean << " \t var = " << var << std::endl;

		for (size_t i=0;i<nObs;i++)
		{
			weight_t &weight = weights[i];
			weight(0) = ( residual_(2*i) - theta * alpha * std::log( std::max(1e-11,(double)residual_(2*i)) ) )     / std::max(1e-11,theta*residual_(2*i)*residual_(2*i));
			weight(1) = ( residual_(2*i+1) - theta * alpha * std::log( std::max(1e-11,(double)residual_(2*i+1)) ) ) / std::max(1e-11, theta*residual_(2*i+1)*residual_(2*i+1));

			// DEBUG
			if (true)
			{
				if( isinf(weight(0)) || isnan(weight(0)) || weight(0) < 0.f )  weight(0) = 0.f;
				if( isinf(weight(1)) || isnan(weight(1)) || weight(1) < 0.f )  weight(1) = 0.f;
			}
			else
			{
				if( isinf(weight(0)) || isnan(weight(0)) )  weight(0) = 1.f;
				if( isinf(weight(1)) || isnan(weight(1)) )  weight(1) = 1.f;
			}

			//f1 << weight(0) << std::endl << weight(1) << std::endl;
			//std::cout << std::endl << weights[i].transpose() << std::endl << std::endl;

			// Update total residual
			residual_t &delta = residuals[i];
			const double sum_2 = weight(0)*residual_(2*i)*residual_(2*i) + weight(1)*residual_(2*i+1)*residual_(2*i+1);

			if (this->parameters.srba.use_robust_kernel)
			{
				const double nrm = std::max(1e-11,std::sqrt(sum_2));
				//const double w = std::sqrt(huber_kernel(nrm,parameters.srba.kernel_param))/nrm;
				const double w = std::sqrt(cauchy_loss_kernel(nrm))/nrm;
				delta *= w;
				total_sqr_err += (w*w)*sum_2;
			}
			else
			{
				// nothing else to do:
				total_sqr_err += sum_2;
			}

		}

	}
	else
	{

		std::cout << std::endl << "Failed Gamma: \t alpha = "  << alpha+1.f << " \t theta = " << theta << " \t mean = " << mean << " \t var = " << var << std::endl;

		stdv      = 1.f;
		for (size_t i=0;i<nObs;i++)
		{
			weight_t &weight = weights[i];
			weight(0) = 1.f;
			weight(1) = 1.f;
			//f1 << weight(0) << std::endl << weight(1) << std::endl;
			// Update total residual
			residual_t &delta = residuals[i];
			const double sum_2 = delta.squaredNorm();
			if (this->parameters.srba.use_robust_kernel)
			{
				const double nrm = std::max(1e-11,std::sqrt(sum_2));
				//const double w = std::sqrt(huber_kernel(nrm,parameters.srba.kernel_param))/nrm;
				const double w = std::sqrt(cauchy_loss_kernel(nrm))/nrm;
				delta *= w;
				total_sqr_err += (w*w)*sum_2;
			}
			else
			{
				// nothing else to do:
				total_sqr_err += sum_2;
			}
		}
	}

	//f.close();   //RGO_DBG
	//f1.close();  //RGO_DBG

	return total_sqr_err;
}

} // End of namespaces
