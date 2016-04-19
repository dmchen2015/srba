/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#pragma once

#include <mrpt/poses/CPose3DQuat.h>

namespace srba {
	/** \defgroup mrpt_srba_models Sensor models: one for each combination of {landmark_parameterization,observation_type}
	  * \ingroup mrpt_srba_grp */

	/** \addtogroup mrpt_srba_models
		* @{ */


	/** Sensor model: 3D landmarks in Euclidean coordinates + Monocular camera observations (no distortion) */
	template <>
	struct sensor_model<landmarks::Euclidean3D,observations::MonocularCamera>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::MonocularCamera  OBS_T;  
		typedef landmarks::Euclidean3D         LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			double x,y,z; // wrt cam (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1],lm_pos[2], x,y,z);
			ASSERT_(z!=0)

			// Pinhole model:
			observation_traits<OBS_T>::array_obs_t  pred_obs;  // prediction
			pred_obs[0] = params.camera_calib.cx() + params.camera_calib.fx() * x/z;
			pred_obs[1] = params.camera_calib.cy() + params.camera_calib.fy() * y/z;
			out_obs_err = z_obs - pred_obs;
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(x_obs);
			// xji_l[0:2]=[X Y Z]
			// If the point is behind us, mark this Jacobian as invalid. This is probably a temporary situation until we get closer to the optimum.
			if (xji_l[2]<=0)
				return false;

			const double pz_inv = 1.0/xji_l[2];
			const double pz_inv2 = pz_inv*pz_inv;

			const double cam_fx = sensor_params.camera_calib.fx();
			const double cam_fy = sensor_params.camera_calib.fy();

			dh_dx.coeffRef(0,0)=  cam_fx * pz_inv;
			dh_dx.coeffRef(0,1)=  0;
			dh_dx.coeffRef(0,2)=  -cam_fx * xji_l[0] * pz_inv2;

			dh_dx.coeffRef(1,0)=  0;
			dh_dx.coeffRef(1,1)=  cam_fy * pz_inv;
			dh_dx.coeffRef(1,2)=  -cam_fy * xji_l[1] * pz_inv2;

			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			//     > +Z 
			//    / 
			//   / 
			//  +----> +X
			//  |
			//  |
			//  V +Y
			//
			const double fx = params.camera_calib.fx();
			const double fy = params.camera_calib.fy();
			const double cx = params.camera_calib.cx();
			const double cy = params.camera_calib.cy();
			out_lm_pos[0] = (obs.px.x-cx)/fx;
			out_lm_pos[1] = (obs.px.y-cy)/fy;
			out_lm_pos[2] = 1; // Depth in Z (undefined for monocular cameras, just use any !=0 value)
		}

	};  // end of struct sensor_model<landmarks::Euclidean3D,observations::MonocularCamera>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 3D line segment landmarks in Euclidean coordinates + Stereo camera observations (no distortion) */
	template <>
	struct sensor_model<landmarks::Euclidean3DLineSegments,observations::StereoCameraLineSegment>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::StereoCameraLineSegment	OBS_T;
		typedef landmarks::Euclidean3DLineSegments      LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;

		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>			TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t   array_landmark_t;    //!< a 2D or 3D point
		typedef OBS_T::TObservationParams						TObservationParams;

		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)"
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t				& out_obs_err,
			const observation_traits<OBS_T>::array_obs_t        & l_obs,
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params )
		{
			double slx,sly,slz, elx,ely,elz; // wrt cam (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1],lm_pos[2], slx,sly,slz);
			ASSERT_(slz!=0)
			base_pose_wrt_observer.composePoint(lm_pos[3],lm_pos[4],lm_pos[5], elx,ely,elz);
			ASSERT_(elz!=0)

			// observed line segments
			Eigen::Vector3f l_obs_l, l_obs_r, spl, epl, spr, epr;
			spl << l_obs[0], l_obs[1], 1.;
			spr << l_obs[2], l_obs[3], 1.;
			epl << l_obs[4], l_obs[5], 1.;
			epr << l_obs[6], l_obs[7], 1.;
			l_obs_l = spl.cross(epl);
			l_obs_l = l_obs_l / std::sqrt( l_obs_l(0)*l_obs_l(0)+l_obs_l(1)*l_obs_l(1) );
			l_obs_r = spr.cross(epr);
			l_obs_r = l_obs_r / std::sqrt( l_obs_r(0)*l_obs_r(0)+l_obs_r(1)*l_obs_r(1) );

			// Pinhole model: Left camera.
			const mrpt::utils::TCamera &lc = params.camera_calib.leftCamera;
			Eigen::Vector2d pred_obs_s, pred_obs_e;
			pred_obs_s << lc.cx() + lc.fx() * slx/slz,
						  lc.cy() + lc.fy() * sly/slz;
			pred_obs_e << lc.cx() + lc.fx() * elx/elz,
						  lc.cy() + lc.fy() * ely/elz;
			out_obs_err[0] = pred_obs_s[0] - spl(0);
			out_obs_err[1] = pred_obs_s[1] - spl(1);
			out_obs_err[4] = pred_obs_e[0] - epl(0);
			out_obs_err[5] = pred_obs_e[1] - epl(1);

			// Project point relative to right-camera:
			const mrpt::poses::CPose3DQuat R2L = -params.camera_calib.rightCameraPose; // R2L = (-) Left-to-right_camera_pose

			// base_wrt_right_cam = ( (-)L2R ) (+) L2Base
			double srx,sry,srz, erx,ery,erz; // wrt cam (local coords)

			// xji_l_right = R2L (+) Xji_l
			R2L.composePoint(slx,sly,slz, srx,sry,srz);
			ASSERT_(srz!=0)
			R2L.composePoint(elx,ely,elz, erx,ery,erz);
			ASSERT_(erz!=0)

			// Pinhole model: Right camera.
			const mrpt::utils::TCamera &rc = params.camera_calib.rightCamera;
			pred_obs_s << rc.cx() + rc.fx() * srx/srz,
						  rc.cy() + rc.fy() * sry/srz;
			pred_obs_e << rc.cx() + rc.fx() * erx/erz,
						  rc.cy() + rc.fy() * ery/erz;
			out_obs_err[2] = pred_obs_s[0] - spr(0);
			out_obs_err[3] = pred_obs_s[1] - spr(1);
			out_obs_err[6] = pred_obs_e[0] - epr(0);
			out_obs_err[7] = pred_obs_e[1] - epr(1);

		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x'
		  *
		  * \endcode
		  *  With:
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  *
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l,
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & l_obs)
		{
			// xji_l[0:2]=[X Y Z]
			// If the point is behind us, mark this Jacobian as invalid. This is probably a temporary situation until we get closer to the optimum.
			if ( xji_l[2]<=0. || xji_l[5]<=0. )
				return false;

			dh_dx = Eigen::Matrix<double,OBS_DIMS,LM_DIMS>::Zero();
			const double fx = sensor_params.camera_calib.leftCamera.fx();
			const double fy = sensor_params.camera_calib.leftCamera.fy();

			// Left camera:
			{
				const double lx = l_obs(0);
				const double ly = l_obs(1);
				// start point (d_pl_P)
				const double pz_inv_s  = 1.0/xji_l[2];
				const double pz_inv2_s = pz_inv_s*pz_inv_s;
				dh_dx.coeffRef(0,0)  = pz_inv_s * fx * lx * lx;
				dh_dx.coeffRef(0,1)  = pz_inv_s * lx * ly * fy;
				dh_dx.coeffRef(0,2)  = -pz_inv2_s * ( lx*lx*fx*xji_l[0] + lx*ly*fy*xji_l[1] );
				dh_dx.coeffRef(1,0)  = pz_inv_s * lx * ly * fx;
				dh_dx.coeffRef(1,1)  = pz_inv_s * ly * ly * fy;
				dh_dx.coeffRef(1,2)  = -pz_inv2_s * ( lx*ly*fx*xji_l[0] + ly*ly*fy*xji_l[1] );
				// end point (d_ql_Q)
				const double pz_inv_e  = 1.0/xji_l[5];
				const double pz_inv2_e = pz_inv_e*pz_inv_e;
				dh_dx.coeffRef(4,3)  = pz_inv_s * fx * lx * lx;
				dh_dx.coeffRef(4,4)  = pz_inv_s * lx * ly * fy;
				dh_dx.coeffRef(4,5)  = -pz_inv2_s * ( lx*lx*fx*xji_l[3] + lx*ly*fy*xji_l[4] );
				dh_dx.coeffRef(5,3)  = pz_inv_s * lx * ly * fx;
				dh_dx.coeffRef(5,4)  = pz_inv_s * ly * ly * fy;
				dh_dx.coeffRef(5,5)  = -pz_inv2_s * ( lx*ly*fx*xji_l[3] + ly*ly*fy*xji_l[4] );
			}

			// Right camera:
			array_landmark_t   xji_l_right_s, xji_l_right_e; // xji_l_right = R2L (+) Xji_l
			const mrpt::poses::CPose3DQuat R2L = -sensor_params.camera_calib.rightCameraPose; // R2L = (-) Left-to-right_camera_pose
			R2L.composePoint(
				xji_l[0],xji_l[1],xji_l[2],
				xji_l_right_s[0],xji_l_right_s[1],xji_l_right_s[2]);
			R2L.composePoint(
				xji_l[3],xji_l[4],xji_l[5],
				xji_l_right_e[0],xji_l_right_e[1],xji_l_right_e[2]);
			{
				const double lx = l_obs(3);
				const double ly = l_obs(4);
				// start point (d_pr_P)
				const double pz_inv_s  = 1.0/xji_l_right_s[2];
				const double pz_inv2_s = pz_inv_s*pz_inv_s;
				dh_dx.coeffRef(2,0)  = pz_inv_s * fx * lx * lx;
				dh_dx.coeffRef(2,1)  = pz_inv_s * lx * ly * fy;
				dh_dx.coeffRef(2,2)  = -pz_inv2_s * ( lx*lx*fx*xji_l_right_s[0] + lx*ly*fy*xji_l_right_s[1] );
				dh_dx.coeffRef(3,0)  = pz_inv_s * lx * ly * fx;
				dh_dx.coeffRef(3,1)  = pz_inv_s * ly * ly * fy;
				dh_dx.coeffRef(3,2)  = -pz_inv2_s * ( lx*ly*fx*xji_l_right_s[0] + ly*ly*fy*xji_l_right_s[1] );
				// end point (d_qr_Q)
				const double pz_inv_e  = 1.0/xji_l_right_e[2];
				const double pz_inv2_e = pz_inv_e*pz_inv_e;
				dh_dx.coeffRef(6,3)  = pz_inv_s * fx * lx * lx;
				dh_dx.coeffRef(6,4)  = pz_inv_s * lx * ly * fy;
				dh_dx.coeffRef(6,5)  = -pz_inv2_s * ( lx*lx*fx*xji_l_right_e[0] + lx*ly*fy*xji_l_right_e[1] );
				dh_dx.coeffRef(7,3)  = pz_inv_s * lx * ly * fx;
				dh_dx.coeffRef(7,4)  = pz_inv_s * ly * ly * fy;
				dh_dx.coeffRef(7,5)  = -pz_inv2_s * ( lx*ly*fx*xji_l_right_e[0] + ly*ly*fy*xji_l_right_e[1] );
			}

			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false"
		  *  in an observation.
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs,
			const OBS_T::TObservationParams               & params)
		{
			//     > +Z
			//    /
			//   /
			//  +----> +X
			//  |
			//  |
			//  V +Y
			//
			const double fxl = params.camera_calib.leftCamera.fx();
			const double fyl = params.camera_calib.leftCamera.fy();
			const double cxl = params.camera_calib.leftCamera.cx();
			const double cyl = params.camera_calib.leftCamera.cy();
			const double baseline  = params.camera_calib.rightCameraPose.x();
			ASSERT_(baseline!=0)
			// start point
			const double disparity_s = std::max(0.001f, obs.start_l_px.x - obs.start_r_px.x);
			const double Z_s = fxl*baseline/disparity_s;
			out_lm_pos[0] = (obs.start_l_px.x - cxl)*Z_s/fxl;
			out_lm_pos[1] = (obs.start_l_px.y - cyl)*Z_s/fyl;
			out_lm_pos[2] = Z_s;
			// end point
			const double disparity_e = std::max(0.001f, obs.end_l_px.x - obs.end_r_px.x);
			const double Z_e = fxl*baseline/disparity_e;
			out_lm_pos[3] = (obs.end_l_px.x - cxl)*Z_e/fxl;
			out_lm_pos[4] = (obs.end_l_px.y - cyl)*Z_e/fyl;
			out_lm_pos[5] = Z_e;
		}

	};  // end of struct sensor_model<landmarks::Euclidean3D,observations::StereoCamera>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 3D landmarks in Euclidean coordinates + Stereo camera observations (no distortion) */
	template <>
	struct sensor_model<landmarks::Euclidean3D,observations::StereoCamera>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::StereoCamera     OBS_T;  
		typedef landmarks::Euclidean3D         LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;

		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			double lx,ly,lz; // wrt cam (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1],lm_pos[2], lx,ly,lz);
			ASSERT_(lz!=0)

			observation_traits<OBS_T>::array_obs_t  pred_obs;  // prediction
			// Pinhole model: Left camera.
			const mrpt::utils::TCamera &lc = params.camera_calib.leftCamera;
			pred_obs[0] = lc.cx() + lc.fx() * lx/lz;
			pred_obs[1] = lc.cy() + lc.fy() * ly/lz;

			// Project point relative to right-camera:
			const mrpt::poses::CPose3DQuat R2L = -params.camera_calib.rightCameraPose; // R2L = (-) Left-to-right_camera_pose
			
			// base_wrt_right_cam = ( (-)L2R ) (+) L2Base
			//mrpt::poses::CPose3DQuat base_wrt_right_cam(mrpt::poses::UNINITIALIZED_POSE);
			//base_wrt_right_cam.composeFrom(R2L,mrpt::poses::CPose3DQuat(base_pose_wrt_observer));
			
			double rx,ry,rz; // wrt cam (local coords)
			//base_wrt_right_cam.composePoint(lx,ly,lz, rx,ry,rz);

			// xji_l_right = R2L (+) Xji_l
			R2L.composePoint(lx,ly,lz, rx,ry,rz);
			ASSERT_(rz!=0)

			// Pinhole model: Right camera.
			const mrpt::utils::TCamera &rc = params.camera_calib.rightCamera;
			pred_obs[2] = rc.cx() + rc.fx() * rx/rz;
			pred_obs[3] = rc.cy() + rc.fy() * ry/rz;
			out_obs_err = z_obs - pred_obs;

			if( abs(out_obs_err[3]) == 1 || abs(out_obs_err[3]) == 1. )
				std::cout << "z_obs = " << z_obs.transpose() << "\t pred_obs = " << pred_obs.transpose() << "\t delta = " << out_obs_err.transpose() << std::endl;

		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(x_obs);
			// xji_l[0:2]=[X Y Z]
			// If the point is behind us, mark this Jacobian as invalid. This is probably a temporary situation until we get closer to the optimum.
			if (xji_l[2]<=0)
				return false;
			
			// Left camera:
			{
				const double pz_inv = 1.0/xji_l[2];
				const double pz_inv2 = pz_inv*pz_inv;

				const double cam_fx = sensor_params.camera_calib.leftCamera.fx();
				const double cam_fy = sensor_params.camera_calib.leftCamera.fy();

				dh_dx.coeffRef(0,0)=  cam_fx * pz_inv;
				dh_dx.coeffRef(0,1)=  0;
				dh_dx.coeffRef(0,2)=  -cam_fx * xji_l[0] * pz_inv2;

				dh_dx.coeffRef(1,0)=  0;
				dh_dx.coeffRef(1,1)=  cam_fy * pz_inv;
				dh_dx.coeffRef(1,2)=  -cam_fy * xji_l[1] * pz_inv2;
			}

			// Right camera:
			array_landmark_t   xji_l_right; // xji_l_right = R2L (+) Xji_l
			const mrpt::poses::CPose3DQuat R2L = -sensor_params.camera_calib.rightCameraPose; // R2L = (-) Left-to-right_camera_pose
			R2L.composePoint(
				xji_l[0],xji_l[1],xji_l[2],
				xji_l_right[0],xji_l_right[1],xji_l_right[2]);
			{
				const double pz_inv = 1.0/xji_l_right[2];
				const double pz_inv2 = pz_inv*pz_inv;

				const double cam_fx = sensor_params.camera_calib.rightCamera.fx();
				const double cam_fy = sensor_params.camera_calib.rightCamera.fy();

				dh_dx.coeffRef(2,0)=  cam_fx * pz_inv;
				dh_dx.coeffRef(2,1)=  0;
				dh_dx.coeffRef(2,2)=  -cam_fx * xji_l_right[0] * pz_inv2;

				dh_dx.coeffRef(3,0)=  0;
				dh_dx.coeffRef(3,1)=  cam_fy * pz_inv;
				dh_dx.coeffRef(3,2)=  -cam_fy * xji_l_right[1] * pz_inv2;
			}

			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			//     > +Z 
			//    / 
			//   / 
			//  +----> +X
			//  |
			//  |
			//  V +Y
			//
			const double fxl = params.camera_calib.leftCamera.fx();
			const double fyl = params.camera_calib.leftCamera.fy();
			const double cxl = params.camera_calib.leftCamera.cx();
			const double cyl = params.camera_calib.leftCamera.cy();
			const double disparity = std::max(0.001f, obs.left_px.x - obs.right_px.x);
			const double baseline  = params.camera_calib.rightCameraPose.x();
			ASSERT_(baseline!=0)
			const double Z = fxl*baseline/disparity;
			out_lm_pos[0] = (obs.left_px.x - cxl)*Z/fxl;
			out_lm_pos[1] = (obs.left_px.y - cyl)*Z/fyl;
			out_lm_pos[2] = Z;
		}

	};  // end of struct sensor_model<landmarks::Euclidean3D,observations::StereoCamera>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 3D landmarks in Euclidean coordinates + Cartesian 3D observations */
	template <>
	struct sensor_model<landmarks::Euclidean3D,observations::Cartesian_3D>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::Cartesian_3D     OBS_T;  
		typedef landmarks::Euclidean3D         LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>        TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams                     TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			MRPT_UNUSED_PARAM(params);
			double x,y,z; // wrt cam (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1],lm_pos[2], x,y,z);

			observation_traits<OBS_T>::array_obs_t  pred_obs;  // prediction
			// Observations are simply the "local coords":
			pred_obs[0] = x; pred_obs[1] = y; pred_obs[2] = z;
			out_obs_err = z_obs - pred_obs;
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(xji_l); MRPT_UNUSED_PARAM(sensor_params); MRPT_UNUSED_PARAM(x_obs);
			// xji_l[0:2]=[X Y Z]
			// This is probably the simplest Jacobian ever:
			dh_dx.setIdentity();
			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			MRPT_UNUSED_PARAM(params);
			out_lm_pos[0] = obs.pt.x;
			out_lm_pos[1] = obs.pt.y;
			out_lm_pos[2] = obs.pt.z;
		}

	};  // end of struct sensor_model<landmarks::Euclidean3D,observations::Cartesian_3D>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 2D landmarks in Euclidean coordinates + Cartesian 2D observations */
	template <>
	struct sensor_model<landmarks::Euclidean2D,observations::Cartesian_2D>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::Cartesian_2D     OBS_T;  
		typedef landmarks::Euclidean2D         LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			double x,y; // wrt cam (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1], x,y);

			observation_traits<OBS_T>::array_obs_t  pred_obs;  // prediction
			// Observations are simply the "local coords":
			pred_obs[0] = x; pred_obs[1] = y;
			out_obs_err = z_obs - pred_obs;
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(xji_l); MRPT_UNUSED_PARAM(sensor_params); MRPT_UNUSED_PARAM(x_obs);
			// xji_l[0:1]=[X Y]
			// This is probably the simplest Jacobian ever:
			dh_dx.setIdentity();
			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			MRPT_UNUSED_PARAM(params);
			out_lm_pos[0] = obs.pt.x;
			out_lm_pos[1] = obs.pt.y;
		}

	};  // end of struct sensor_model<landmarks::Euclidean2D,observations::Cartesian_2D>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 3D landmarks in Euclidean coordinates + 3D Range-Bearing observations */
	template <>
	struct sensor_model<landmarks::Euclidean3D,observations::RangeBearing_3D>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::RangeBearing_3D  OBS_T;  
		typedef landmarks::Euclidean3D         LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			mrpt::math::TPoint3D  l; // wrt sensor (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1],lm_pos[2], l.x,l.y,l.z);
			
			static const mrpt::poses::CPose3D origin;
			double range,yaw,pitch;
			origin.sphericalCoordinates(
				l,  // In: point
				range,yaw,pitch // Out: spherical coords							
				);

			observation_traits<OBS_T>::array_obs_t  pred_obs;  // prediction
			pred_obs[0] = range;
			pred_obs[1] = yaw;
			pred_obs[2] = pitch;
			out_obs_err = z_obs - pred_obs;
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(sensor_params); MRPT_UNUSED_PARAM(x_obs);
			// xji_l[0:2]=[X Y Z]
			mrpt::math::CMatrixDouble33 dh_dx_(mrpt::math::UNINITIALIZED_MATRIX);

			static const mrpt::poses::CPose3DQuat origin;
			double range,yaw,pitch;
			origin.sphericalCoordinates(
				mrpt::math::TPoint3D(xji_l[0],xji_l[1],xji_l[2]),  // In: point
				range,yaw,pitch, // Out: spherical coords
				&dh_dx_,   // dh_dx
				NULL       // dh_dp (not needed)
				);

			dh_dx = dh_dx_;
			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			MRPT_UNUSED_PARAM(params);
			const double chn_y = cos(obs.yaw),   shn_y = sin(obs.yaw);
			const double chn_p = cos(obs.pitch), shn_p = sin(obs.pitch);
			// The new point, relative to the sensor:
			out_lm_pos[0] =  obs.range * chn_y * chn_p;
			out_lm_pos[1] =  obs.range * shn_y * chn_p;
			out_lm_pos[2] = -obs.range * shn_p;
		}

	};  // end of struct sensor_model<landmarks::Euclidean3D,observations::RangeBearing_3D>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 2D landmarks in Euclidean coordinates + 2D Range-Bearing observations */
	template <>
	struct sensor_model<landmarks::Euclidean2D,observations::RangeBearing_2D>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::RangeBearing_2D  OBS_T;  
		typedef landmarks::Euclidean2D         LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err,
			const observation_traits<OBS_T>::array_obs_t        & z_obs,
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			MRPT_UNUSED_PARAM(params);
			mrpt::math::TPoint2D  l; // wrt sensor (local coords)
			base_pose_wrt_observer.composePoint(lm_pos[0],lm_pos[1], l.x,l.y);
			
			const double range = hypot(l.x,l.y);
			const double yaw   = atan2(l.y,l.x);

			observation_traits<OBS_T>::array_obs_t  pred_obs;  // prediction
			pred_obs[0] = range;
			pred_obs[1] = yaw;
			out_obs_err = z_obs - pred_obs;
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l,
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(sensor_params); MRPT_UNUSED_PARAM(x_obs);
			// xji_l[0:1]=[X Y]
			const double r = hypot(xji_l[0], xji_l[1]);
			if (r==0) return false;

			const double r_inv = 1.0/r;
			const double r_inv2 = r_inv*r_inv;
			dh_dx(0,0) = xji_l[0] * r_inv;
			dh_dx(0,1) = xji_l[1] * r_inv;
			dh_dx(1,0) = -xji_l[1] * r_inv2;
			dh_dx(1,1) =  xji_l[0] * r_inv2;
			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			MRPT_UNUSED_PARAM(params);
			const double chn_y = cos(obs.yaw), shn_y = sin(obs.yaw);
			// The new point, relative to the sensor:
			out_lm_pos[0] = obs.range * chn_y;
			out_lm_pos[1] = obs.range * shn_y;
		}

	}; // end of sensor_model<landmarks::Euclidean2D,observations::RangeBearing_2D>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 2D relative poses landmarks and observations (relative 2D-graph SLAM) */
	template <>
	struct sensor_model<landmarks::RelativePoses2D,observations::RelativePoses_2D>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::RelativePoses_2D  OBS_T;  
		typedef landmarks::RelativePoses2D      LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			MRPT_UNUSED_PARAM(lm_pos); MRPT_UNUSED_PARAM(params);
			// Relative pose observation: 
			//  OUT_OBS_ERR = - pseudo-log( PREDICTED_REL_POSE \ominus Z_OBS )
			const POSE_T h = POSE_T(z_obs[0],z_obs[1],z_obs[2]) - base_pose_wrt_observer;

			out_obs_err[0] = h.x();
			out_obs_err[1] = h.y();
			out_obs_err[2] = h.phi();
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(xji_l); MRPT_UNUSED_PARAM(sensor_params); MRPT_UNUSED_PARAM(x_obs);
			// h(z_obs \ominus p) = pseudo-log(z_obs \ominus  p)
			// with p: relative pose in SE(2)
			dh_dx.setIdentity();
			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			MRPT_UNUSED_PARAM(params);
			out_lm_pos[0] = obs.x;
			out_lm_pos[1] = obs.y;
			out_lm_pos[2] = obs.yaw;
		}

	}; // end of sensor_model<landmarks::RelativePoses2D,observations::RelativePoses_2D>

	// -------------------------------------------------------------------------------------------------------------

	/** Sensor model: 3D relative poses landmarks and observations (relative 3D-graph SLAM) */
	template <>
	struct sensor_model<landmarks::RelativePoses3D,observations::RelativePoses_3D>
	{
		// --------------------------------------------------------------------------------
		// Typedefs for the sake of generality in the signature of methods below:
		//   *DONT FORGET* to change these when writing new sensor models.
		// --------------------------------------------------------------------------------
		typedef observations::RelativePoses_3D  OBS_T;  
		typedef landmarks::RelativePoses3D      LANDMARK_T;
		// --------------------------------------------------------------------------------

		static const size_t OBS_DIMS = OBS_T::OBS_DIMS;
		static const size_t LM_DIMS  = LANDMARK_T::LM_DIMS;
		
		typedef Eigen::Matrix<double,OBS_DIMS,LM_DIMS>  TJacobian_dh_dx;     //!< A Jacobian of the correct size for each dh_dx
		typedef landmark_traits<LANDMARK_T>::array_landmark_t    array_landmark_t;             //!< a 2D or 3D point
		typedef OBS_T::TObservationParams               TObservationParams;


		/** Executes the (negative) observation-error model: "-( h(lm_pos,pose) - z_obs)" 
		  * \param[out] out_obs_err The output of the predicted sensor value
		  * \param[in] z_obs The real observation, to be contrasted to the prediction of this sensor model
		  * \param[in] base_pose_wrt_observer The relative pose of the observed landmark's base KF, wrt to the current sensor pose (which may be different than the observer KF pose if the sensor is not at the "robot origin").
		  * \param[in] lm_pos The relative landmark position wrt its base KF.
		  * \param[in] params The sensor-specific parameters.
		  */
		template <class POSE_T>
		static void observe_error(
			observation_traits<OBS_T>::array_obs_t              & out_obs_err, 
			const observation_traits<OBS_T>::array_obs_t        & z_obs, 
			const POSE_T                                        & base_pose_wrt_observer,
			const landmark_traits<LANDMARK_T>::array_landmark_t & lm_pos,
			const OBS_T::TObservationParams                     & params)
		{
			// Relative pose observation: 
			//  OUT_OBS_ERR = - pseudo-log( PREDICTED_REL_POSE \ominus Z_OBS )
			const POSE_T h = POSE_T(z_obs[0],z_obs[1],z_obs[2],z_obs[3],z_obs[4],z_obs[5]) - base_pose_wrt_observer;

			mrpt::poses::SE_traits<3>::pseudo_ln(h,out_obs_err);
		}

		/** Evaluates the partial Jacobian dh_dx:
		  * \code
		  *            d h(x')
		  * dh_dx = -------------
		  *             d x' 
		  *
		  * \endcode
		  *  With: 
		  *    - x' = x^{j,i}_l  The relative location of the observed landmark wrt to the robot/camera at the instant of observation. (See notation on papers)
		  *    - h(x): Observation model: h(): landmark location --> observation
		  * 
		  * \param[out] dh_dx The output matrix Jacobian. Values at input are undefined (i.e. they cannot be asssumed to be zeros by default).
		  * \param[in]  xji_l The relative location of the observed landmark wrt to the robot/camera at the instant of observation.
		  * \param[in] sensor_params Sensor-specific parameters, as set by the user.
		  *
		  * \return true if the Jacobian is well-defined, false to mark it as ill-defined and ignore it during this step of the optimization
		  */
		static bool eval_jacob_dh_dx(
			TJacobian_dh_dx          & dh_dx,
			const array_landmark_t   & xji_l, 
			const TObservationParams & sensor_params,
			const observation_traits<OBS_T>::array_obs_t & x_obs)
		{
			MRPT_UNUSED_PARAM(xji_l); MRPT_UNUSED_PARAM(sensor_params); MRPT_UNUSED_PARAM(x_obs);
			// h(z_obs \ominus p) = xji_l (the relative pose in SE(3)) 
			dh_dx.setIdentity();
			return true;
		}

		/** Inverse observation model for first-seen landmarks. Needed to avoid having landmarks at (0,0,0) which 
		  *  leads to undefined Jacobians. This is invoked only when both "unknown_relative_position_init_val" and "is_fixed" are "false" 
		  *  in an observation. 
		  * The LM location must not be exact at all, just make sure it doesn't have an undefined Jacobian.
		  *
		  * \param[out] out_lm_pos The relative landmark position wrt the current observing KF.
		  * \param[in]  obs The observation itself.
		  * \param[in]   params The sensor-specific parameters.
		  */
		static void inverse_sensor_model(
			landmark_traits<LANDMARK_T>::array_landmark_t & out_lm_pos,
			const observation_traits<OBS_T>::obs_data_t   & obs, 
			const OBS_T::TObservationParams               & params)
		{
			MRPT_UNUSED_PARAM(params);
			out_lm_pos[0] = obs.x;   out_lm_pos[1] = obs.y;     out_lm_pos[2] = obs.z;
			out_lm_pos[3] = obs.yaw; out_lm_pos[4] = obs.pitch; out_lm_pos[5] = obs.roll; 
		}

	};  // end of struct sensor_model<landmarks::RelativePoses3D,observations::RelativePoses_3D>

	// -------------------------------------------------------------------------------------------------------------

	/** @} */

} // end NS
