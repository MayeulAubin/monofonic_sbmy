#pragma once

#include <array>
#include <vector>

#include <general.hh>

#include <math/vec3.hh>

template <int interp_order, typename grid_t>
struct grid_interpolate
{
  using data_t = typename grid_t::data_t;
  using vec3 = std::array<real_t, 3>;

  static constexpr bool is_distributed_trait = grid_t::is_distributed_trait;
  static constexpr int interpolation_order = interp_order;

  
#if defined(USE_MPI)
  const MPI_Datatype MPI_data_t_type =
      (typeid(data_t) == typeid(float)) ? MPI_FLOAT
                                        : (typeid(data_t) == typeid(double)) ? MPI_DOUBLE
                                                                             : (typeid(data_t) == typeid(long double)) ? MPI_LONG_DOUBLE
                                                                                                                       : (typeid(data_t) == typeid(std::complex<float>)) ? MPI_C_FLOAT_COMPLEX
                                                                                                                                                                         : (typeid(data_t) == typeid(std::complex<double>)) ? MPI_C_DOUBLE_COMPLEX
                                                                                                                                                                                                                            : (typeid(data_t) == typeid(std::complex<long double>)) ? MPI_C_LONG_DOUBLE_COMPLEX
                                                                                                                                                                                                                                                                                    : MPI_INT;
#endif

  std::vector<data_t> boundary_;
  const grid_t &gridref;
  size_t nx_, ny_, nz_;

  explicit grid_interpolate(const grid_t &g)
      : gridref(g), nx_(g.n_[0]), ny_(g.n_[1]), nz_(g.n_[2])
  {
    static_assert(interpolation_order >= 0 && interpolation_order <= 2, "Interpolation order needs to be 0 (NGP), 1 (CIC), or 2 (TSC).");

    if (is_distributed_trait)
    {
#if defined(USE_MPI)
      size_t nx = interpolation_order + 1;
      size_t ny = g.n_[1];
      size_t nz = g.n_[2];

      boundary_.assign(nx * ny * nz, data_t{0.0});

      for (size_t i = 0; i < nx; ++i)
      {
        for (size_t j = 0; j < ny; ++j)
        {
          for (size_t k = 0; k < nz; ++k)
          {
            boundary_[(i * ny + j) * nz + k] = g.relem(i, j, k);
          }
        }
      }

      int sendto = (MPI::get_rank() + MPI::get_size() - 1) % MPI::get_size();
      int recvfrom = (MPI::get_rank() + MPI::get_size() + 1) % MPI::get_size();

      MPI_Status status;
      status.MPI_ERROR = MPI_SUCCESS;

      int err = MPI_Sendrecv_replace(&boundary_[0], nx * ny * nz, MPI::get_datatype<data_t>(), sendto,
                           MPI::get_rank() + 1000, recvfrom, recvfrom + 1000, MPI_COMM_WORLD, &status);

      if( err != MPI_SUCCESS ){
        char errstr[256]; int errlen=256;
        MPI_Error_string(err, errstr, &errlen ); 
        music::elog << "MPI_ERROR #" << err << " : " << errstr << std::endl;
      }

#endif
    }
  }

  data_t get_ngp_at(const std::array<real_t, 3> &pos, std::vector<data_t> &val) const noexcept
  {
    size_t ix = static_cast<size_t>(pos[0]);
    size_t iy = static_cast<size_t>(pos[1]);
    size_t iz = static_cast<size_t>(pos[2]);
    return gridref.relem(ix - gridref.local_0_start_, iy, iz);
  }

  data_t get_cic_at(const std::array<real_t, 3> &pos) const noexcept
  {
    size_t ix = static_cast<size_t>(pos[0]);
    size_t iy = static_cast<size_t>(pos[1]);
    size_t iz = static_cast<size_t>(pos[2]);
    real_t dx = pos[0] - real_t(ix), tx = 1.0 - dx;
    real_t dy = pos[1] - real_t(iy), ty = 1.0 - dy;
    real_t dz = pos[2] - real_t(iz), tz = 1.0 - dz;
    size_t iy1 = (iy + 1) % ny_;
    size_t iz1 = (iz + 1) % nz_;

    data_t val{0.0};
    
    if( is_distributed_trait ){
      ptrdiff_t localix = ix-gridref.local_0_start_;
      val += gridref.relem(localix, iy, iz) * tx * ty * tz;
      val += gridref.relem(localix, iy, iz1) * tx * ty * dz;
      val += gridref.relem(localix, iy1, iz) * tx * dy * tz;
      val += gridref.relem(localix, iy1, iz1) * tx * dy * dz;

      if( localix+1 >= gridref.local_0_size_ ){
        size_t localix1 = localix+1 - gridref.local_0_size_;
        val += boundary_[(localix1*ny_+iy)*nz_+iz] * dx * ty * tz;
        val += boundary_[(localix1*ny_+iy)*nz_+iz1] * dx * ty * dz;
        val += boundary_[(localix1*ny_+iy1)*nz_+iz] * dx * dy * tz;
        val += boundary_[(localix1*ny_+iy1)*nz_+iz1] * dx * dy * dz;
      }else{
        size_t localix1 = localix+1;
        val += gridref.relem(localix1, iy, iz) * dx * ty * tz;
        val += gridref.relem(localix1, iy, iz1) * dx * ty * dz;
        val += gridref.relem(localix1, iy1, iz) * dx * dy * tz;
        val += gridref.relem(localix1, iy1, iz1) * dx * dy * dz;
      }
    }else{
      size_t ix1 = (ix + 1) % nx_;
      val += gridref.relem(ix, iy, iz) * tx * ty * tz;
      val += gridref.relem(ix, iy, iz1) * tx * ty * dz;
      val += gridref.relem(ix, iy1, iz) * tx * dy * tz;
      val += gridref.relem(ix, iy1, iz1) * tx * dy * dz;
      val += gridref.relem(ix1, iy, iz) * dx * ty * tz;
      val += gridref.relem(ix1, iy, iz1) * dx * ty * dz;
      val += gridref.relem(ix1, iy1, iz) * dx * dy * tz;
      val += gridref.relem(ix1, iy1, iz1) * dx * dy * dz;
    }
    return val;
  }

  // data_t get_tsc_at(const std::array<real_t, 3> &pos, std::vector<data_t> &val) const
  // {
  // }

  int get_task(const vec3 &x, const std::vector<int> &local0starts) const noexcept
  {
    const auto it = std::upper_bound(local0starts.begin(), local0starts.end(), int(x[0]));
    return std::distance(local0starts.begin(), it)-1;
  }

  void domain_decompose_pos(std::vector<vec3> &pos) const noexcept
  {
    if (is_distributed_trait)
    {
#if defined(USE_MPI)
      int local_0_start = int(gridref.local_0_start_);
      std::vector<int> local0starts(MPI::get_size(), 0);
      MPI_Alltoall(&local_0_start, 1, MPI_INT, &local0starts[0], 1, MPI_INT, MPI_COMM_WORLD);

      std::sort(pos.begin(), pos.end(), [&](auto x1, auto x2) { return get_task(x1,local0starts) < get_task(x2,local0starts); });
      std::vector<int> sendcounts(MPI::get_size(), 0), sendoffsets(MPI::get_size(), 0);
      std::vector<int> recvcounts(MPI::get_size(), 0), recvoffsets(MPI::get_size(), 0);
      for (auto x : pos)
      {
        sendcounts[get_task(x,local0starts)] += 3;
      }

      MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, MPI_COMM_WORLD);

      size_t tot_receive = recvcounts[0], tot_send = sendcounts[0];
      for (int i = 1; i < MPI::get_size(); ++i)
      {
        sendoffsets[i] = sendcounts[i - 1] + sendoffsets[i - 1];
        recvoffsets[i] = recvcounts[i - 1] + recvoffsets[i - 1];
        tot_receive += recvcounts[i];
        tot_send += sendcounts[i];
      }

      std::vector<vec3> recvbuf;
      recvbuf.assign(tot_receive,{0.,0.,0.});

      MPI_Alltoallv(&pos[0], &sendcounts[0], &sendoffsets[0], MPI_data_t_type,
                    &recvbuf[0], &recvcounts[0], &recvoffsets[0], MPI_data_t_type, MPI_COMM_WORLD);

      std::swap( pos, recvbuf );
#endif
    }
  }

  ccomplex_t compensation_kernel( const vec3_t<real_t>& k ) const noexcept
  {
    auto sinc = []( real_t x ){ return (std::abs(x)>1e-10)? std::sin(x)/x : 1.0; };
    real_t dfx = sinc(0.5*M_PI*k[0]/gridref.kny_[0]);
    real_t dfy = sinc(0.5*M_PI*k[1]/gridref.kny_[1]);
    real_t dfz = sinc(0.5*M_PI*k[2]/gridref.kny_[2]);
    real_t del = std::pow(dfx*dfy*dfz,1+interpolation_order);

    real_t shift = 0.5 * k[0] * gridref.get_dx()[0] + 0.5 * k[1] * gridref.get_dx()[1] + 0.5 * k[2] * gridref.get_dx()[2];

    return std::exp(ccomplex_t(0.0, shift)) / del;
  }

  void get_at(std::vector<vec3> &pos, std::vector<data_t> &val) const
  {

    val.assign( pos.size(), data_t{0.0} );

    for( size_t i=0; i<pos.size(); ++i ){
      const vec3& x = pos[i];

      switch (interpolation_order)
      {
      case 0:
        val[i] = get_ngp_at(x);
        break;
      case 1:
        val[i] = get_cic_at(x);
        break;
      // case 2:
      //   val[i] = get_tsc_at(x);
      //   break;
      };
    }
  }
};