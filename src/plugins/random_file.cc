// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
//
// RNG plugin "FILE": read a real-space white-noise field from an HDF5 file
// (e.g. a pysbmy/Simbelmyne field file, data set /scalars/field, C order, axis 0 = x)
// instead of generating it. Each MPI rank reads only its own x-slab by hyperslab, so
// HDF5 virtual data sets over per-tile files work too.
//
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <general.hh>
#include <random_plugin.hh>
#include <config_file.hh>
#include <grid_fft.hh>
#include <HDF_IO.hh>

#include <stdexcept>

class RNG_file : public RNG_plugin
{
private:
    std::string fname_, dset_;

public:
    explicit RNG_file(config_file &cf) : RNG_plugin(cf)
    {
        fname_ = cf.get_value<std::string>("random", "noise_file");
        dset_ = cf.get_value_safe<std::string>("random", "noise_dataset", "/scalars/field");
    }

    bool isMultiscale() const { return false; }

    // Same convention as MUSIC1: unit-variance real-space noise; the unitary forward FFT in
    // ic_generator.cc then gives <|w_k|^2> = 1, as NGENIC fills Fourier space directly.
    void Fill_Grid(Grid_FFT<real_t> &g)
    {
        g.FourierTransformBackward(false);
        g.zero();

        hid_t file = H5Fopen(fname_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
            throw std::runtime_error("FILE noise: cannot open '" + fname_ + "'");
        hid_t dset = H5Dopen2(file, dset_.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            H5Fclose(file);
            throw std::runtime_error("FILE noise: no data set '" + dset_ + "' in '" + fname_ + "'");
        }
        hid_t fspace = H5Dget_space(dset);
        hsize_t dims[3] = {0, 0, 0};
        const int ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims == 3)
            H5Sget_simple_extent_dims(fspace, dims, nullptr);
        if (ndims != 3 || dims[0] != g.n_[0] || dims[1] != g.n_[1] || dims[2] != g.n_[2])
        {
            music::elog << "FILE noise: '" << fname_ << ":" << dset_ << "' has shape " << dims[0] << "x" << dims[1] << "x"
                        << dims[2] << " but setup/GridRes is " << g.n_[0] << std::endl;
            H5Sclose(fspace); H5Dclose(dset); H5Fclose(file);
            throw std::runtime_error("FILE noise: grid size mismatch");
        }

        // file: this rank's x-slab; memory: the FFTW real-space array, padded along z to npr_
        const hsize_t nloc = g.local_0_size_;
        hsize_t fstart[3] = {hsize_t(g.local_0_start_), 0, 0}, count[3] = {nloc, dims[1], dims[2]};
        hsize_t mdims[3] = {std::max<hsize_t>(nloc, 1), dims[1], hsize_t(g.npr_)}, mstart[3] = {0, 0, 0};
        hid_t mspace = H5Screate_simple(3, mdims, nullptr);
        if (nloc > 0)
        {
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, fstart, nullptr, count, nullptr);
            H5Sselect_hyperslab(mspace, H5S_SELECT_SET, mstart, nullptr, count, nullptr);
        }
        else
        {
            H5Sselect_none(fspace);
            H5Sselect_none(mspace);
        }
        // HDF5 converts float32/float64 on disk to real_t
        herr_t status = H5Dread(dset, GetDataType<real_t>(), mspace, fspace, H5P_DEFAULT, g.data_);
        H5Sclose(mspace); H5Sclose(fspace); H5Dclose(dset); H5Fclose(file);
        if (status < 0)
            throw std::runtime_error("FILE noise: read failed for '" + fname_ + "'");

        double s1 = 0.0, s2 = 0.0;
        for (size_t i = 0; i < nloc; ++i)
            for (size_t j = 0; j < dims[1]; ++j)
                for (size_t k = 0; k < dims[2]; ++k)
                {
                    const double v = g.relem(i, j, k);
                    s1 += v; s2 += v * v;
                }
        const double nn = double(nloc) * dims[1] * dims[2];
        if (nn > 0)
            music::ilog << "FILE noise: read " << fname_ << ":" << dset_ << ", local slab mean = " << s1 / nn
                        << ", rms = " << std::sqrt(s2 / nn) << " (expect 0, 1)" << std::endl;
    }
};

namespace
{
RNG_plugin_creator_concrete<RNG_file> creator("FILE");
}
