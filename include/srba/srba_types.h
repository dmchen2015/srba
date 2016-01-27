/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#pragma once

#include <mrpt/math/lightweight_geom_data.h>
#include <mrpt/math/MatrixBlockSparseCols.h>
#include <mrpt/math/CArrayNumeric.h>
#include <mrpt/utils/TEnumType.h>
#include <mrpt/system/memory.h> // for MRPT_MAKE_ALIGNED_OPERATOR_NEW
#include <set>

namespace srba
{
	typedef uint64_t  TKeyFrameID; //!< Numeric IDs for key-frames (KFs)
	typedef uint64_t  TLandmarkID; //!< Numeric IDs for landmarks
	typedef uint64_t  topo_dist_t;  //!< Unsigned integral type for topological distances in a graph/tree.
	typedef std::pair<TKeyFrameID,TKeyFrameID> TPairKeyFrameID;  //!< Used to represent the IDs of a directed edge (first --> second)
	typedef std::multimap<size_t,TKeyFrameID,std::greater<size_t> > base_sorted_lst_t;  //!< A list of keyframes, sorted in descending order by some count number

	#define VERBOSE_LEVEL(_LEVEL) if (m_verbose_level>=_LEVEL) std::cout

	#define VERBOSE_LEVEL_COLOR(_LEVEL,_TEXT_COLOR) if (m_verbose_level>=_LEVEL) { mrpt::system::setConsoleColor(_TEXT_COLOR); std::cout
	#define VERBOSE_LEVEL_COLOR_POST() mrpt::system::setConsoleColor(mrpt::system::CONCOL_NORMAL); }

	#define SRBA_INVALID_KEYFRAMEID  static_cast<TKeyFrameID>(-1)
	#define SRBA_INVALID_INDEX       static_cast<size_t>(-1)

	// -------------------------------------------------------------------------------
	/** @name Generic traits for observations, kf-to-kf poses, landmarks, etc.
	    @{ */

	/** Generic declaration, of which specializations are defined for each combination of LM+OBS type.
	  * \sa Implementations are in srba/models/sensors.h
	  */
	template <class landmark_t,class obs_t>
	struct sensor_model;

	/** The argument "POSE_TRAITS" can be any of those defined in srba/models/kf2kf_poses.h (typically, either kf2kf_poses::SE3 or kf2kf_poses::SE2).
	  * \sa landmark_traits, observation_traits
	  */
	template <class POSE_TRAITS>
	struct kf2kf_pose_traits: public POSE_TRAITS
	{
		typedef kf2kf_pose_traits<POSE_TRAITS> me_t;
		typedef typename POSE_TRAITS::pose_t  pose_t;  //!< Will be mrpt::poses::CPose3D, ...
		typedef typename mrpt::math::CArrayDouble<POSE_TRAITS::REL_POSE_DIMS>  array_pose_t;  //!< A fixed-length array of the size of the relative poses between keyframes

		/** A joint structure for one relative pose + an "up-to-date" flag (needed for spanning trees numeric updates) */
		struct pose_flag_t
		{
			pose_t        pose;
			mutable bool  updated;

			pose_flag_t() : updated(false) { }
			pose_flag_t(const pose_t &pose_, const bool updated_) : pose(pose_),updated(updated_) { }

			inline void mark_outdated() const { updated=false; }

			MRPT_MAKE_ALIGNED_OPERATOR_NEW  // Needed because we have fixed-length Eigen matrices (within CPose3D)
		};

		/**  "numeric" values of spanning tree poses: */
		typedef typename mrpt::aligned_containers<TKeyFrameID, pose_flag_t>::map_t  frameid2pose_map_t;

		// Use special map-like container with an underlying planar deque container, which avoid reallocations
		// and still provides map-like [] access at O(1).
		typedef mrpt::utils::map_as_vector<
			TKeyFrameID,
			frameid2pose_map_t,
			typename std::deque<std::pair<TKeyFrameID,frameid2pose_map_t> >
			>  TRelativePosesForEachTarget;

		/** Keyframe-to-keyframe edge: an unknown of the problem */
		struct k2k_edge_t
		{
			TKeyFrameID  from;
			TKeyFrameID  to;
			pose_t       inv_pose; //!< Inverse pose: pose_from (-) pose_to , that is: "from" as seen from "to".

			size_t       id; //!< 0-based index of this edge, in the std::list "k2k_edges".

			MRPT_MAKE_ALIGNED_OPERATOR_NEW    // Needed because we have fixed-length Eigen matrices (within CPose3D)
		};

		typedef std::deque<k2k_edge_t*>  k2k_edge_vector_t; //!< A sequence of edges (a "path")
	}; // end of kf2kf_pose_traits

	/** The argument "LM_TRAITS" can be any of those defined in srba/models/landmarks.h (typically, either landmarks::Euclidean3D or landmarks::Euclidean2D).
	  * \sa landmark_traits, observation_traits
	  */
	template <class LM_TRAITS>
	struct landmark_traits: public LM_TRAITS
	{
		typedef landmark_traits<LM_TRAITS> me_t;
		typedef typename mrpt::math::CArrayDouble<LM_TRAITS::LM_DIMS>        array_landmark_t;  //!< A fixed-length array of the size of the parameters of one landmark in the map (e.g. if Euclidean coordinates are used, this will hold the coordinates)

		/** One relative feature observation entry, used with some relative bundle-adjustment functions. */
		struct TRelativeLandmarkPos
		{
			inline TRelativeLandmarkPos() { }
			/** Constructor from a base KF ID "_id_frame_base" and any object "_pos" that offers a read [] operator and has the correct length of "LM_TRAITS::LM_DIMS"
			  *  \tparam LANDMARK_POS Could be: "array_landmark_t", "double *", "mrpt::math::TPoint3D", etc.
			  */
			template <typename LANDMARK_POS>
			inline TRelativeLandmarkPos(const TKeyFrameID  _id_frame_base, const LANDMARK_POS &_pos) : id_frame_base(_id_frame_base) {
				for (size_t i=0;i<LM_TRAITS::LM_DIMS;i++) pos[i]=_pos[i];
			}

			TKeyFrameID  id_frame_base;	//!< The ID of the camera frame which is the coordinate reference of \a pos
			array_landmark_t   pos;  //!< The parameterization of the feature location, wrt to the camera frame \a id_frame_base - For example, this could simply be Euclidean coordinates (x,y,z)

			MRPT_MAKE_ALIGNED_OPERATOR_NEW  // Needed because we have fixed-length Eigen matrices ("array_landmark_t")
		};

		/** An index of feature IDs and their relative locations */
		typedef std::map<TLandmarkID, TRelativeLandmarkPos>  TRelativeLandmarkPosMap;

		/** Used in the vector \a "all_lms" */
		struct TLandmarkEntry
		{
			bool                 has_known_pos; //!< true: This landmark has a fixed (known) relative position. false: The relative pos of this landmark is an unknown of the problem.
			TRelativeLandmarkPos *rfp;           //!< Pointers to elements in \a unknown_lms and \a known_lms.

			TLandmarkEntry() : has_known_pos(true), rfp(NULL) {}
			TLandmarkEntry(bool has_known_pos_, TRelativeLandmarkPos *rfp_) : has_known_pos(has_known_pos_), rfp(rfp_)
			{}
		};

	}; // end of landmark_traits

	template <class OBS_TRAITS>
	struct observation_traits : public OBS_TRAITS
	{
		typedef typename mrpt::math::CArrayDouble<OBS_TRAITS::OBS_DIMS>  array_obs_t;  //!< A fixed-length array of the size of one residual (=the size of one observation).
		typedef typename mrpt::math::CArrayDouble<OBS_TRAITS::OBS_DIMS>  residual_t;   //!< A fixed-length array of the size of one residual (=the size of one observation).
        typedef typename mrpt::math::CArrayDouble<OBS_TRAITS::OBS_DIMS/2>  weight_t;   //!< A fixed-length array of 2x the size of one residual (=0.5x the size of one observation).

		typedef typename mrpt::aligned_containers<residual_t>::vector_t  vector_residuals_t;
        typedef typename mrpt::aligned_containers<weight_t>::vector_t    vector_weights_t;

		/** Elemental observation data */
		struct observation_t
		{
			TLandmarkID   feat_id;  //!< Observed what
			typename OBS_TRAITS::obs_data_t  obs_data; //!< Observed data
		};

		MRPT_MAKE_ALIGNED_OPERATOR_NEW

	}; // end of "observation_traits"

	/** @} */
	// --------------------------------------------------------------

	/** Covariances recovery from Hessian matrix policy, for usage in RbaEngine.parameters
	  *  \note Implements mrpt::utils::TEnumType
	  */
	enum TCovarianceRecoveryPolicy {
		/** Don't recover any covariance information */
		crpNone = 0,
		/** Approximate covariances of landmarks as the inverse of the hessian diagonal blocks */
		crpLandmarksApprox
	};

	/** Aux function for getting, from a std::pair, "the other" element which is different to a known given one. */
	template <typename PAIR,typename V>
	V getTheOtherFromPair(const V one, const PAIR &p) {
		return p.first==one ? p.second : p.first;
	}

	/** For usage with K2K_EDGE = typename kf2kf_pose_traits<POSE_TRAITS>::k2k_edge_t */
	template <typename K2K_EDGE,typename V>
	V getTheOtherFromPair2(const V one, const K2K_EDGE &p) {
		return p.from==one ? p.to: p.from;
	}

	/** Used in TNewKeyFrameInfo */
	struct TNewEdgeInfo
	{
		size_t  id; //!< The new edge ID
		bool    has_approx_init_val; //!< Whether the edge was assigned an approximated initial value. If not, it will need an independent optimization step before getting into the complete problem optimization.

		/** For loop closures situations, and only for helping estimating an initial value for the new edge (when \a has_approx_init_val is false), 
		  * these fields hold the IDs of KFs from which to infer the relative pose using the sensor model.
		  * Filled in ECP's eval<>() method, used determine_kf2kf_edges_to_create() */
		TKeyFrameID loopclosure_observer_kf, loopclosure_base_kf;

		TNewEdgeInfo() :
			id                      (SRBA_INVALID_INDEX),
			has_approx_init_val     (false),
			loopclosure_observer_kf (SRBA_INVALID_KEYFRAMEID),
			loopclosure_base_kf     (SRBA_INVALID_KEYFRAMEID)
		{}
	};

	/** Symbolic information of each Jacobian dh_dAp
	  */
	template <class kf2kf_pose_t, class LANDMARK_TYPE>
	struct TJacobianSymbolicInfo_dh_dAp
	{
		typedef kf2kf_pose_traits<kf2kf_pose_t> kf2kf_traits_t;
		typedef landmark_traits<LANDMARK_TYPE>     lm_traits_t;

		/** The two relative poses used in this Jacobian (see papers)
		  * Pointers to the elements in the "numeric" part of the spanning tree ( TRBA_Problem_state::TSpanningTree ) */
		const typename kf2kf_traits_t::pose_flag_t * rel_pose_d1_from_obs, * rel_pose_base_from_d1;

		/** If true, the edge direction points in the "canonical" direction: from "d" towards the direction of "obs"
		  *  If false, this fact should be taking into account while computing the derivatives... */
		bool  edge_normal_dir;

		/** Pointer to the relative feature position wrt its base KF */
		const typename lm_traits_t::TRelativeLandmarkPos * feat_rel_pos;

		/** The ID of the keyframe *before* the edge wrt which we are taking derivatives (before if going
		  *  backwards from the observer KF towards the base KF of the feature). "d+1" in paper figures */
		TKeyFrameID  kf_d;
		TKeyFrameID  kf_base;  //!< The ID of the base keyframe of the observed feature
		size_t       k2k_edge_id; //!< The ID (0-based index in \a k2k_edges) of the edge wrt we are taking derivatives
		size_t       obs_idx; //!< The global index of the observation that generates this Jacobian
		char       * is_valid; //!< A reference to the validity bit in the global list \a TRBA_Problem_state::all_observations_Jacob_validity

		TJacobianSymbolicInfo_dh_dAp() : 
			rel_pose_d1_from_obs(NULL),
			rel_pose_base_from_d1(NULL),
			edge_normal_dir(true),
			feat_rel_pos(NULL),
			kf_d(SRBA_INVALID_KEYFRAMEID),
			kf_base(SRBA_INVALID_KEYFRAMEID),
			k2k_edge_id(SRBA_INVALID_INDEX),
			obs_idx(SRBA_INVALID_INDEX),
			is_valid(NULL)
		{}
	};


	/** Symbolic information of each Jacobian dh_df
	  */
	template <class kf2kf_pose_t, class LANDMARK_TYPE>
	struct TJacobianSymbolicInfo_dh_df
	{
		typedef kf2kf_pose_traits<kf2kf_pose_t> kf2kf_traits_t;
		typedef landmark_traits<LANDMARK_TYPE>     lm_traits_t;

		typename lm_traits_t::TRelativeLandmarkPos  *feat_rel_pos;  //!< A pointer to the relative position structure within rba_state.unknown_lms[] for this feature

		/** The relative poses used in this Jacobian (see papers)
		  * Pointers to the elements in the "numeric" part of the spanning tree ( TRBA_Problem_state::TSpanningTree ) */
		const typename kf2kf_traits_t::pose_flag_t * rel_pose_base_from_obs;

		size_t   obs_idx;  //!< The global index of the observation that generates this Jacobian
		char   * is_valid; //!< A reference to the validity bit in the global list \a TRBA_Problem_state::all_observations_Jacob_validity

		TJacobianSymbolicInfo_dh_df() :
			feat_rel_pos(NULL),
			rel_pose_base_from_obs(NULL),
			obs_idx(SRBA_INVALID_INDEX),
			is_valid(NULL)
		{}
	};

	/** Symbolic information of each Hessian block
	  *  \tparam Scalar Typ.=double
	  *  \tparam N Observation size
	  *  \tparam M1 Left-hand Jacobian column count
	  *  \tparam M2 Right-hand Jacobian column count
	  */
	template <typename Scalar,int N,int M1,int M2>
	struct THessianSymbolicInfo
	{
		typedef Eigen::Matrix<Scalar,N,M1> matrix1_t;
		typedef Eigen::Matrix<Scalar,N,M2> matrix2_t;

		/** This Hessian block equals the sum of all J1^t * \Lambda * J2, with J1=first, J2=second in each std::pair
		  * "const char *" are pointers to the validity bit of each Jacobian, so if it evaluates to false we
		  * should discard the Hessian entry.
		  */
		struct THessianSymbolicInfoEntry
		{
			const matrix1_t * J1; 
			const matrix2_t * J2; 
			const char *      J1_valid;
			const char *      J2_valid;
			size_t            obs_idx; //!< Global index of the observation that generated this Hessian entry (used to retrieve the \Lambda in "J1^t * \Lambda * J2", if applicable).

			THessianSymbolicInfoEntry(const matrix1_t * const J1_, const matrix2_t * const J2_, const char * const J1_valid_, const char * const J2_valid_, const size_t obs_idx_ ) :
				J1(J1_), J2(J2_),J1_valid(J1_valid_),J2_valid(J2_valid_),obs_idx(obs_idx_) 
			{ }

			// Default ctor: should not be invoked under normal usage, but just in case:
			THessianSymbolicInfoEntry() : J1(NULL),J2(NULL),J1_valid(NULL),J2_valid(NULL),obs_idx(static_cast<size_t>(-1)) {}
		};

		typedef std::vector<THessianSymbolicInfoEntry> list_jacob_blocks_t;

		list_jacob_blocks_t lst_jacob_blocks; //!< The list of Jacobian blocks itself
	};


	/** A wrapper of mrpt::math::MatrixBlockSparseCols<> with extended functionality. Refer to the docs of the base class. 
	  * A templated column-indexed efficient storage of block-sparse Jacobian or Hessian matrices, together with other arbitrary information.
	  * Columns are stored in a non-associative container, but the contents of each column are kept within an std::map<> indexed by row.
	  * All submatrix blocks have the same size, which allows dense storage of them in fixed-size matrices, avoiding costly memory allocations.
	  */
	template <
		typename Scalar,
		int NROWS,
		int NCOLS,
		typename INFO,
		bool HAS_REMAP,
		typename INDEX_REMAP_MAP_IMPL = mrpt::utils::map_as_vector<size_t,size_t>
	>
	struct SparseBlockMatrix : public mrpt::math::MatrixBlockSparseCols<Scalar,NROWS,NCOLS,INFO,HAS_REMAP,INDEX_REMAP_MAP_IMPL>
	{
		typedef mrpt::math::MatrixBlockSparseCols<Scalar,NROWS,NCOLS,INFO,HAS_REMAP,INDEX_REMAP_MAP_IMPL> base_t;

		/** Goes over all the columns and keep the largest span MAX_ROW_INDEX - MIN_ROW_INDEX, the column length */
		size_t findRowSpan(size_t *row_min_idx=NULL, size_t *row_max_idx=NULL) const
		{
			size_t row_min=std::numeric_limits<size_t>::max();
			size_t row_max=0;
			const size_t nCols = base_t::getColCount();
			for (size_t j=0;j<nCols;j++)
				for (typename base_t::col_t::const_iterator itRow=base_t::getCol(j).begin();itRow!=base_t::getCol(j).end();++itRow) {
					mrpt::utils::keep_max(row_max, itRow->first);
					mrpt::utils::keep_min(row_min, itRow->first);
				}
			if (row_min==std::numeric_limits<size_t>::max())
				row_min=0;
			if (row_min_idx) *row_min_idx = row_min;
			if (row_max_idx) *row_max_idx = row_max;
			return (row_max-row_min)+1;
		}

		//! \overload For a subset of columns
		static size_t findRowSpan(const std::vector<typename base_t::col_t*> &lstColumns, size_t *row_min_idx=NULL, size_t *row_max_idx=NULL)
		{
			size_t row_min=std::numeric_limits<size_t>::max();
			size_t row_max=0;
			for (size_t i=0;i<lstColumns.size();++i)
				for (typename base_t::col_t::const_iterator itRow=lstColumns[i]->begin();itRow!=lstColumns[i]->end();++itRow) {
					mrpt::utils::keep_max(row_max, itRow->first);
					mrpt::utils::keep_min(row_min, itRow->first);
				}
			if (row_min==std::numeric_limits<size_t>::max())
				row_min=0;
			if (row_min_idx) *row_min_idx = row_min;
			if (row_max_idx) *row_max_idx = row_max;
			return (row_max-row_min)+1;
		}

		/** Returns the number of (symbolically) non-zero blocks, etc. 
			* \param[out] nMaxBlocks The maximum number of block matrices fitting in this sparse matrix if it was totally dense
			* \param[out] nNonZeroBlocks The number of non-zero blocks
			*/
		void getSparsityStats(size_t &nMaxBlocks, size_t &nNonZeroBlocks  ) const {
			const size_t nCols = base_t::getColCount();
			const size_t nRows = findRowSpan();
			nMaxBlocks = nCols * nRows;
			nNonZeroBlocks = 0;
			for (size_t j=0;j<nCols;j++)
				nNonZeroBlocks+= base_t::getCol(j).size();
		}

		//! \overload For a subset of columns
		static void getSparsityStats(const std::vector<typename base_t::col_t*> &lstColumns, size_t &nMaxBlocks, size_t &nNonZeroBlocks  )
		{
			const size_t nCols = lstColumns.size();
			const size_t nRows = findRowSpan(lstColumns);
			nMaxBlocks = nCols * nRows;
			nNonZeroBlocks = 0;
			for (size_t j=0;j<nCols;j++)
				nNonZeroBlocks+=lstColumns[j]->size();
		}

	}; // end SparseBlockMatrix()


	/** Types for the Jacobians:
	  * \code
	  *   J = [  dh_dAp  |  dh_df ]
	  * \endcode
	  */
	template <class kf2kf_pose_t, class LANDMARK_TYPE,class obs_t>
	struct jacobian_traits
	{
		static const size_t OBS_DIMS      = obs_t::OBS_DIMS;
		static const size_t REL_POSE_DIMS = kf2kf_pose_t::REL_POSE_DIMS;
		static const size_t LM_DIMS       = LANDMARK_TYPE::LM_DIMS;

		typedef TJacobianSymbolicInfo_dh_dAp<kf2kf_pose_t,LANDMARK_TYPE> jacob_dh_dAp_info_t;
		typedef TJacobianSymbolicInfo_dh_df<kf2kf_pose_t,LANDMARK_TYPE>  jacob_dh_df_info_t;

		typedef SparseBlockMatrix<double,OBS_DIMS,REL_POSE_DIMS,jacob_dh_dAp_info_t, false>  TSparseBlocksJacobians_dh_dAp;  //!< The "false" is since we don't need to "remap" indices
		typedef SparseBlockMatrix<double,OBS_DIMS,LM_DIMS,jacob_dh_df_info_t,  true >   TSparseBlocksJacobians_dh_df;  // The "true" is to "remap" indices
	};

	/** Types for the Hessian blocks:
	  * \code
	  *       [  H_Ap    |  H_Apf  ]
	  *   H = [ ---------+-------- ]
	  *       [  H_Apf^t |   Hf    ]
	  * \endcode
	  */
	template <class kf2kf_pose_t, class LANDMARK_TYPE,class obs_t>
	struct hessian_traits
	{
		static const size_t OBS_DIMS      = obs_t::OBS_DIMS;
		static const size_t REL_POSE_DIMS = kf2kf_pose_t::REL_POSE_DIMS;
		static const size_t LM_DIMS       = LANDMARK_TYPE::LM_DIMS;

		typedef THessianSymbolicInfo<double,OBS_DIMS,REL_POSE_DIMS,REL_POSE_DIMS> hessian_Ap_info_t;
		typedef THessianSymbolicInfo<double,OBS_DIMS,LM_DIMS,LM_DIMS>             hessian_f_info_t;
		typedef THessianSymbolicInfo<double,OBS_DIMS,REL_POSE_DIMS,LM_DIMS>       hessian_Apf_info_t;

		// (the final "false" in all types is because we don't need remapping of indices in hessians)
		typedef SparseBlockMatrix<double,REL_POSE_DIMS , REL_POSE_DIMS , hessian_Ap_info_t , false> TSparseBlocksHessian_Ap;
		typedef SparseBlockMatrix<double,LM_DIMS       , LM_DIMS       , hessian_f_info_t  , false> TSparseBlocksHessian_f;
		typedef SparseBlockMatrix<double,REL_POSE_DIMS , LM_DIMS       , hessian_Apf_info_t, false> TSparseBlocksHessian_Apf;

		/** The list with all the information matrices (estimation uncertainty) for each unknown landmark. */
		typedef mrpt::utils::map_as_vector<
			TLandmarkID,
			typename TSparseBlocksHessian_f::matrix_t,
			typename mrpt::aligned_containers<std::pair<TLandmarkID,typename TSparseBlocksHessian_f::matrix_t > >::deque_t> landmarks2infmatrix_t;
	};


	/** Useful data structures that depend of a combination of "OBSERVATION_TYPE"+"LANDMARK_PARAMETERIZATION_TYPE"+"RELATIVE_POSE_PARAMETERIZATION"
	  */
	template <class kf2kf_pose_t,class landmark_t,class obs_t>
	struct rba_joint_parameterization_traits_t
	{
		typedef landmark_t   original_landmark_t;
		typedef kf2kf_pose_t original_kf2kf_pose_t;

		typedef kf2kf_pose_traits<kf2kf_pose_t> kf2kf_traits_t;
		typedef observation_traits<obs_t>       obs_traits_t;
		typedef landmark_traits<landmark_t>           lm_traits_t;

		typedef typename kf2kf_traits_t::k2k_edge_t k2k_edge_t;

		/** Observations, as provided by the user. The following combinations are possible:
		  *  \code
		  *  +-----------+---------------------------------------------+---------------------------------+
		  *  |                  DATA FIELDS                            |                                 |
		  *  +-----------+-------------------------+-------------------+       RESULTING OBSERVATION     |
		  *  | is_fixed  | is_unknown_with_init_val|      feat_rel_pos      |                                 |
		  *  +-----------+-------------------------+-------------------+---------------------------------+
		  *  |           |                         |                   | First observation of a landmark |
		  *  |   true    |     ( IGNORED )         | Landmark position |  with a fixed (known) relative  |
		  *  |           |                         |                   |    position wrt this keyframe   |
		  *  +-----------+-------------------------+-------------------+---------------------------------+
		  *  |           |                         |                   | First observation of a landmark |
		  *  |   false   |         true            | Landmark position |  with unknown relative position |
		  *  |           |                         |                   | whose guess is given in feat_rel_pos |
		  *  +-----------+-------------------------+-------------------+---------------------------------+
		  *  |           |                         |                   | Either:                         |
		  *  |           |                         |                   |  * First observation of a LM    |
		  *  |   false   |        false            |     (IGNORED)     | with unknown relative pos.      |
		  *  |           |                         |                   | In this case we'll call  sensor_model<>::inverse_sensor_model()
		  *  |           |                         |                   |  * Subsequent observations of   |
		  *  |           |                         |                   | any fixed or unknown LM.        |
		  *  +-----------+-------------------------+-------------------+---------------------------------+
		  *  \endcode
		  */
		struct new_kf_observation_t
		{
			/** Default ctor */
			new_kf_observation_t() : is_fixed(false), is_unknown_with_init_val(false) { feat_rel_pos.setZero(); }

			typename obs_traits_t::observation_t  obs;

			/** If true, \a feat_rel_pos has the fixed relative position of this landmark (Can be set to true only upon the FIRST observation of a fixed landmark)
			  */
			bool is_fixed;

			/** Can be set to true only upon FIRST observation of a landmark with UNKNOWN relative position (the normal case).
			  *  If set to true, \a feat_rel_pos has the fixed relative position of this landmark.
			  */
			bool is_unknown_with_init_val;

			typename lm_traits_t::array_landmark_t feat_rel_pos; //!< Ignored unless \a is_fixed OR \a is_unknown_with_init_val are true (only one of them at once).

			/** Sets \a feat_rel_pos from any object that offers a [] operator and has the expected length "landmark_t::LM_DIMS" */
			template <class REL_POS> inline void setRelPos(const REL_POS &pos) {
				for (size_t i=0;i<landmark_t::LM_DIMS;i++) feat_rel_pos[i]=pos[i];
			}
		};

		/** A set of all the observations made from a new KF, as provided by the user */
		typedef std::deque<new_kf_observation_t> new_kf_observations_t;

		/** Keyframe-to-feature edge: observations in the problem */
		struct kf_observation_t
		{
			inline kf_observation_t() {}
			inline kf_observation_t(const typename obs_traits_t::observation_t &obs_, const TKeyFrameID kf_id_) : obs(obs_), kf_id(kf_id_) {}

			typename obs_traits_t::observation_t obs;      //!< Observation data
			typename obs_traits_t::array_obs_t   obs_arr;  //!< Observation data, summarized as an array of its parameters:  obs.obs_data.getAsArray(obs_arr);
			TKeyFrameID   kf_id;    //!< Observed from

			MRPT_MAKE_ALIGNED_OPERATOR_NEW  // This forces aligned mem allocation
		};


		/** Keyframe-to-feature edge: observation data stored for each keyframe */
		struct k2f_edge_t
		{
			kf_observation_t  obs;
			bool              feat_has_known_rel_pos;   //!< whether it's a known or unknown relative position feature
			bool              is_first_obs_of_unknown;  //!< true if this is the first observation of a feature with unknown relative position
			typename lm_traits_t::TRelativeLandmarkPos *feat_rel_pos; //!< Pointer to the known/unknown rel.pos. (always!=NULL)

			inline const TLandmarkID get_observed_feature_id() const { return obs.obs.feat_id; }

			MRPT_MAKE_ALIGNED_OPERATOR_NEW  // This forces aligned mem allocation
		};

		/** Information per key-frame needed for RBA */
		struct keyframe_info
		{
			std::deque<k2k_edge_t*>  adjacent_k2k_edges;
			std::deque<k2f_edge_t*>  adjacent_k2f_edges;
		};

	}; // end of "rba_joint_parameterization_traits_t"


	/** Used in TRBA_Problem_state */
	struct TSpanTreeEntry
	{
		TKeyFrameID next;     //!< The next keyframe in the tree
		topo_dist_t distance; //!< Remaining distance until the given target from this point.
	};

	/** All the important data of a RBA problem at any given instant of time
	  *  Operations on this structure are performed via the public API of srba::RbaEngine
	  * \sa RbaEngine
	  */
	template <class kf2kf_pose_t,class landmark_t,class obs_t,class RBA_OPTIONS>
	struct TRBA_Problem_state
	{
		typedef typename kf2kf_pose_t::pose_t pose_t;
		typedef typename kf2kf_pose_traits<kf2kf_pose_t>::k2k_edge_t         k2k_edge_t;
		typedef typename kf2kf_pose_traits<kf2kf_pose_t>::k2k_edge_vector_t  k2k_edge_vector_t;
		typedef typename kf2kf_pose_traits<kf2kf_pose_t>::frameid2pose_map_t frameid2pose_map_t;
		typedef typename kf2kf_pose_traits<kf2kf_pose_t>::pose_flag_t        pose_flag_t;
		typedef typename landmark_traits<landmark_t>::TRelativeLandmarkPosMap TRelativeLandmarkPosMap;
		typedef typename landmark_traits<landmark_t>::TLandmarkEntry          TLandmarkEntry;
		typedef typename hessian_traits<kf2kf_pose_t,landmark_t,obs_t>::landmarks2infmatrix_t   landmarks2infmatrix_t;
		typedef typename rba_joint_parameterization_traits_t<kf2kf_pose_t,landmark_t,obs_t>::keyframe_info          keyframe_info;
		typedef typename rba_joint_parameterization_traits_t<kf2kf_pose_t,landmark_t,obs_t>::k2f_edge_t             k2f_edge_t;
		typedef typename rba_joint_parameterization_traits_t<kf2kf_pose_t,landmark_t,obs_t>::new_kf_observations_t  new_kf_observations_t;
		typedef typename rba_joint_parameterization_traits_t<kf2kf_pose_t,landmark_t,obs_t>::new_kf_observation_t   new_kf_observation_t;

		typedef typename mrpt::aligned_containers<k2k_edge_t>::deque_t  k2k_edges_deque_t;  // Note: A std::deque() does not invalidate pointers/references, as we always insert elements at the end; we'll exploit this...
		typedef typename mrpt::aligned_containers<k2f_edge_t>::deque_t  all_observations_deque_t;

		typedef std::deque<keyframe_info>  keyframe_vector_t;  //!< Index are "TKeyFrameID" IDs. There's no NEED to make this a deque<> for preservation of references, but is an efficiency improvement

		struct TSpanningTree
		{
			/** The definition seems complex but behaves just like: std::map< TKeyFrameID, std::map<TKeyFrameID,TSpanTreeEntry> > */
			typedef mrpt::utils::map_as_vector<
				TKeyFrameID,
				std::map<TKeyFrameID,TSpanTreeEntry>,
				std::deque<std::pair<TKeyFrameID,std::map<TKeyFrameID,TSpanTreeEntry> > >
				> next_edge_maps_t;

			/** The definition seems complex but behaves just like: std::map< TKeyFrameID, std::map<TKeyFrameID, k2k_edge_vector_t> > */
			typedef mrpt::utils::map_as_vector<
				TKeyFrameID,
				std::map<TKeyFrameID, k2k_edge_vector_t>,
				std::deque<std::pair<TKeyFrameID,std::map<TKeyFrameID, k2k_edge_vector_t > > >
				> all_edges_maps_t;

			const TRBA_Problem_state<kf2kf_pose_t,landmark_t,obs_t,RBA_OPTIONS> *m_parent;

			/** @name Data structures
			  *  @{ */

			/** The "symbolic" part of the spanning tree */
			struct TSpanningTreeSym
			{
				/** Given all interesting spanning tree roots (=SOURCE), this structure holds:
				  *   map[SOURCE] |-> map[TARGET] |-> next node to follow
				  *
				  * \note This defines a table with symmetric distances D(i,j)=D(j,i) but with asymetric next nodes N(i,j)!=N(j,i).
				  *        So, we store both [i][j] and [j][i] in all cases.
				  */
				next_edge_maps_t  next_edge;

				/** From the previous data, we can build this alternative, more convenient representation:
				  *   map[SOURCE] |-> map[TARGET] |-> vector of edges to follow.
				  *
				  * \note This table is symmetric since the shortest path i=>j is the same (in reverse order) than that for j=>i.
				  *        So, we only store the entries for [i][j], i>j.
				  */
				all_edges_maps_t all_edges;
			}
			sym;

			/** "Numeric" spanning tree: the SE(3) pose of each node wrt to any other:
			  *   num[SOURCE] |--> map[TARGET] = CPose3D of TARGET as seen from SOURCE
			  *   (typ: SOURCE is the observing KF, TARGET is the reference base of the observed landmark)
			  *
			  *  Numeric poses are valid after calling \a update_numeric()
			  *
			  *  NOTE: Both symmetric poses, e.g. (i,j) and also (j,i), are stored for convenience of
			  *         being able to get references/pointers to them.
			  */
			typename kf2kf_pose_traits<kf2kf_pose_t>::TRelativePosesForEachTarget num;

			/** @} */


			/** @name Spanning tree main operations
			  *  @{ */

			/** Empty all sym & num data */
			void clear();

			/** Incremental update of spanning trees after the insertion of ONE new node and ONE OR MORE edges
			  * \param[in] max_distance Is the maximum distance at which a neighbor can be so we store the shortest path to it.
			  */
			void update_symbolic_new_node(
				const TKeyFrameID                    new_node_id,
				const TPairKeyFrameID & new_edge,
				const topo_dist_t                    max_depth
				);

			/** Updates all the numeric SE(3) poses from ALL the \a sym.all_edges
			  * \return The number of updated poses.
			  */
			size_t update_numeric(bool skip_marked_as_uptodate = false);

			/** idem, for the set of edges that have as "from" node any of the IDs in the passed set. */
			size_t update_numeric(const std::set<TKeyFrameID> & kfs_to_update,bool skip_marked_as_uptodate = false);

			/** Updates all the numeric SE(3) poses from a given entry from \a sym.all_edges[i]
			  * \return The number of updated poses.
			  */
			size_t update_numeric_only_all_from_node( const typename all_edges_maps_t::const_iterator & it,bool skip_marked_as_uptodate = false);

			/** @} */

			/** @name Spanning tree misc. operations
			  *  @{ */

			/** Useful for debugging */
			void dump_as_text(std::string &s) const;

			/** Useful for debugging \return false on error */
			bool dump_as_text_to_file(const std::string &sFileName) const;

			/** Saves all (or a subset of all) the spanning trees
			  * If kf_roots_to_save is left empty, all STs are saved. Otherwise, only those with the given roots.
			  * \return false on error
			  */
			bool save_as_dot_file(const std::string &sFileName, const std::vector<TKeyFrameID> &kf_roots_to_save = std::vector<TKeyFrameID>() ) const;

			/** Returns min/max and mean/std stats on the number of nodes found on all the spanning trees. Runs in O(N), N=number of keyframes. */
			void get_stats(
				size_t &num_nodes_min,
				size_t &num_nodes_max,
				double &num_nodes_mean,
				double &num_nodes_std) const;

			/** @} */

		}; // end of TSpanningTree


		struct TLinearSystem
		{
			TLinearSystem() :
				dh_dAp(),
				dh_df()
			{
			}

			typename jacobian_traits<kf2kf_pose_t,landmark_t,obs_t>::TSparseBlocksJacobians_dh_dAp dh_dAp;   //!< Both symbolic & numeric info on the sparse Jacobians wrt. the edges
			typename jacobian_traits<kf2kf_pose_t,landmark_t,obs_t>::TSparseBlocksJacobians_dh_df  dh_df;    //!< Both symbolic & numeric info on the sparse Jacobians wrt. the observations

			void clear() {
				dh_dAp.clearAll();
				dh_df.clearAll();
			}

		}; // end of TLinearSystem

		/** @name Data
		    @{ */

		keyframe_vector_t       keyframes;   //!< All key frames (global poses are not included in an RBA problem). Vector indices are "TKeyFrameID" IDs.
		k2k_edges_deque_t       k2k_edges;   //!< (unknowns) All keyframe-to-keyframe edges
		TRelativeLandmarkPosMap unknown_lms; //!< (unknown values) Landmarks with an unknown fixed 3D position relative to their base frame_id
		landmarks2infmatrix_t   unknown_lms_inf_matrices; //!< Information matrices that model the uncertainty in each XYZ position for the unknown LMs - these matrices should be already scaled according to the camera noise in pixel standard deviations.
		TRelativeLandmarkPosMap known_lms;   //!< (known values) Landmarks with a known, fixed 3D position relative to their base frame_id

		/** Index (by feat ID) of ALL landmarks stored in \a unknown_lms and \a known_lms.
		  * Note that if gaps occur in the observed feature IDs, some pointers here will be NULL and some mem will be wasted, but in turn we have a O(1) search mechanism for all LMs. */
		typename mrpt::aligned_containers<TLandmarkEntry>::deque_t   all_lms;

		TSpanningTree            spanning_tree;
		all_observations_deque_t all_observations;  //!< All raw observation data (k2f edges)
		TLinearSystem            lin_system;        //!< The sparse linear system of equations

		/** Its size grows simultaneously to all_observations, its values are updated during optimization to
		  *  reflect invalid conditions in some Jacobians so we can completely discard their information
		  *  while building the Hessian.
		  * \note Kept as deque so we can have references to this.
		  */
		std::deque<char>       all_observations_Jacob_validity;

		/** List of KFs touched by new KF2KF edges in the previous timesteps. Used in determine_kf2kf_edges_to_create() to bootstrap initial relative poses. */
		std::set<size_t>       last_timestep_touched_kfs;  
		/** @} */

		/** Empties all members */
		void clear() {
			keyframes.clear();
			k2k_edges.clear();
			unknown_lms.clear();
			unknown_lms_inf_matrices.clear();
			known_lms.clear();
			all_lms.clear();
			spanning_tree.clear();
			all_observations.clear();
			lin_system.clear();
			last_timestep_touched_kfs.clear();
		}

		/** Ctor */
		TRBA_Problem_state() {
			spanning_tree.m_parent=this; // Not passed as ctor argument to avoid compiler warnings...
		}

		/** Auxiliary, brute force (BFS) method for finding the shortest path between any two Keyframes.
		  * Use only when the distance between nodes can be larger than the maximum depth of incrementally-built spanning trees
		  * \param[in,out] out_path_IDs (Ignored if ==NULL) Just leave this vector uninitialized at input, it'll be automatically initialized to the right size and values.
		  * \param[in,out] out_path_edges (Ignored if ==NULL) Just like out_path_IDs, but here you'll receive the list of traversed edges, instead of the IDs of the visited KFs.
		  * \return false if no path was found.
		  */
		bool find_path_bfs(
			const TKeyFrameID           from,
			const TKeyFrameID           to,
			std::vector<TKeyFrameID>  * out_path_IDs,
			typename kf2kf_pose_traits<kf2kf_pose_t>::k2k_edge_vector_t * out_path_edges = NULL) const;


		/** Computes stats on the degree (# of adjacent nodes) of all the nodes in the graph. Runs in O(N) with N=# of keyframes */
		void compute_all_node_degrees(
			double &out_mean_degree,
			double &out_std_degree,
			double &out_max_degree) const;

		/** Returns true if the pair of KFs are connected thru a kf2kf edge, no matter the direction of the edge. Runs in worst-case O(D) with D the degree of the KF graph (that is, the maximum number of edges adjacent to one KF) */
		bool are_keyframes_connected(const TKeyFrameID id1, const TKeyFrameID id2) const;

		/** Creates a new kf2kf edge variable. Called from create_kf2kf_edge()
		  *
		  * \param[in] init_inv_pose_val The initial value for the inverse pose stored in edge first->second, i.e. the pose of first wrt. second.
		  * \return The ID of the new kf2kf edge, which coincides with the 0-based index of the new entry in "rba_state.k2k_edges"
		  *
		  * \note Runs in O(1)
		  */
		size_t alloc_kf2kf_edge(
			const TPairKeyFrameID &ids,
			const pose_t &init_inv_pose_val = pose_t() );

	private:
		// Forbid making copies of this object, since it heavily relies on internal lists of pointers:
		TRBA_Problem_state(const TRBA_Problem_state &);
		TRBA_Problem_state & operator =(const TRBA_Problem_state &);

	}; // end of TRBA_Problem_state

} // end of namespace "srba"

// Specializations MUST occur at the same namespace:
namespace mrpt { namespace utils
{
	template <>
	struct TEnumTypeFiller<srba::TCovarianceRecoveryPolicy>
	{
		typedef srba::TCovarianceRecoveryPolicy enum_t;
		static void fill(bimap<enum_t,std::string>  &m_map)
		{
			m_map.insert(srba::crpNone,      "crpNone");
			m_map.insert(srba::crpLandmarksApprox,   "crpLandmarksApprox");
		}
	};
} } // End of namespace
