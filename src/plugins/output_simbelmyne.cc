#include <unistd.h> // for unlink

#include "HDF_IO.hh"
#include <logger.hh>
#include <output_plugin.hh>

class simbelmyne_output_plugin : public output_plugin
{
    struct gadget4_header_t
  {
    size_t npart[6];
    double mass[6];
    double time;
    double redshift;
    int flag_sfr;
    int flag_feedback;
    size_t npartTotal[6];
    int flag_cooling;
    int num_files;
    double BoxSize;
    double Omega0;
    double OmegaLambda;
    double HubbleParam;
    int flag_stellarage;
    int flag_metals;
    unsigned int npartTotalHighWord[6];
    int flag_entropy_instead_u;
    int flag_doubleprecision;
  };

private:
    std::string get_field_name( const cosmo_species &s, const fluid_component &c );
    template< typename T > void write_header_attribute( const std::string Filename, const std::string ObjName, const T &Data );
    void add_simbelmyne_metadata( const std::string &fname );
    void move_dataset_in_hdf5( const std::string &fname, const std::string &src_dset_name, const std::string &group_name, const std::string &tg_dset_name );
    void write_gadget_header();
    int get_species_idx(const cosmo_species &s) const;

protected:
    bool out_eulerian_;
    bool out_particles_;

    // Gadget variables
    int num_files_, num_simultaneous_writers_;
    gadget4_header_t header_;
    real_t lunit_, vunit_, munit_;
    std::string this_fname_;

public:
    //! constructor
    explicit simbelmyne_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc )
    : output_plugin(cf, pcc, "Simbelmyne HDF5")
    {
        // out_eulerian_   = cf_.get_value_safe<bool>("output", "simbelmyne_out_eulerian", false);
        out_eulerian_ = false;
        out_particles_ = true;

        // Gadget constructor
        num_files_ = 1;
        #ifdef USE_MPI
            // use as many output files as we have MPI tasks
            MPI_Comm_size(MPI_COMM_WORLD, &num_files_);
        #endif
            real_t astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
            const double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3
        
            lunit_ = cf_.get_value<double>("setup", "BoxLength");
            vunit_ = lunit_ / std::sqrt(astart);
            munit_ = rhoc * std::pow(cf_.get_value<double>("setup", "BoxLength"), 3); // in 1e10 h^-1 M_sol
        
            num_simultaneous_writers_ = cf_.get_value_safe<int>("output", "NumSimWriters", num_files_);
    
        
            for (int i = 0; i < 6; ++i)
            {
            header_.npart[i] = 0;
            header_.npartTotal[i] = 0;
            header_.npartTotalHighWord[i] = 0;
            header_.mass[i] = 0.0;
            }
        
            header_.time = astart;
            header_.redshift = 1.0 / astart - 1.0;
            header_.flag_sfr = 0;
            header_.flag_feedback = 0;
            header_.flag_cooling = 0;
            header_.num_files = num_files_;
            header_.BoxSize = lunit_;
            header_.Omega0 = pcc->cosmo_param_["Omega_m"];
            header_.OmegaLambda = pcc->cosmo_param_["Omega_DE"];
            header_.HubbleParam = pcc->cosmo_param_["h"];
            header_.flag_stellarage = 0;
            header_.flag_metals = 0;
            header_.flag_entropy_instead_u = 0;
            header_.flag_doubleprecision = false;
        
        
            this_fname_ = fname_ + "particles";
        #ifdef USE_MPI
            int thisrank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
            if (num_files_ > 1)
            this_fname_ += "." + std::to_string(thisrank);
        #endif
            this_fname_ += ".hdf5";
        
            unlink(this_fname_.c_str());
            HDFCreateFile(this_fname_);
    }

    output_type write_species_as( const cosmo_species &s ) const
    { 
        if( out_eulerian_ )
            return output_type::field_eulerian;
        else if (out_particles_)
            return output_type::particles;
        else
            return output_type::field_lagrangian;
    }

    bool has_64bit_reals() const{ return false; }

    bool has_64bit_ids() const{ return true; }

    real_t position_unit() const { return lunit_; }

    real_t velocity_unit() const { return vunit_; }
  
    real_t mass_unit() const { return munit_; }

    void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c );

    void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species);
};


// Gadget plugin functions
template <typename T>
std::vector<T> from_6array(const T *a)
{
  return std::vector<T>{{a[0], a[1], a[2], a[3], a[4], a[5]}};
}

template <typename T>
std::vector<T> from_value(const T a)
{
  return std::vector<T>{{a}};
}

void simbelmyne_output_plugin::write_gadget_header()
{
    HDFCreateGroup(this_fname_, "Header");
    HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_ThisFile", from_6array<size_t>(header_.npart));
    HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_Total", from_6array<size_t>(header_.npartTotal));
    HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_Total_HighWord", from_6array<unsigned>(header_.npartTotalHighWord));
    HDFWriteGroupAttribute(this_fname_, "Header", "MassTable", from_6array<double>(header_.mass));
    HDFWriteGroupAttribute(this_fname_, "Header", "Time", from_value<double>(header_.time));
    HDFWriteGroupAttribute(this_fname_, "Header", "Redshift", from_value<double>(header_.redshift));
    HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Sfr", from_value<int>(header_.flag_sfr));
    HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Feedback", from_value<int>(header_.flag_feedback));
    HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Cooling", from_value<int>(header_.flag_cooling));
    HDFWriteGroupAttribute(this_fname_, "Header", "NumFilesPerSnapshot", from_value<int>(header_.num_files));
    HDFWriteGroupAttribute(this_fname_, "Header", "BoxSize", from_value<double>(header_.BoxSize));
    HDFWriteGroupAttribute(this_fname_, "Header", "Omega0", from_value<double>(header_.Omega0));
    HDFWriteGroupAttribute(this_fname_, "Header", "OmegaLambda", from_value<double>(header_.OmegaLambda));
    HDFWriteGroupAttribute(this_fname_, "Header", "HubbleParam", from_value<double>(header_.HubbleParam));
    HDFWriteGroupAttribute(this_fname_, "Header", "Flag_StellarAge", from_value<int>(header_.flag_stellarage));
    HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Metals", from_value<int>(header_.flag_metals));
    HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Entropy_ICs", from_value<int>(header_.flag_entropy_instead_u));

    music::ilog << "Wrote Gadget-HDF5 file(s) to " << this_fname_ << std::endl;

    music::ilog << "You can use the following values in param.txt:" << std::endl;
    music::ilog << "Omega0       " << header_.Omega0 << std::endl;
    music::ilog << "OmegaLambda  " << header_.OmegaLambda << std::endl;
    music::ilog << "OmegaBaryon  " << pcc_->cosmo_param_["Omega_b"] << std::endl;
    music::ilog << "HubbleParam  " << header_.HubbleParam << std::endl;
    music::ilog << "Hubble       100.0" <<  std::endl;
    music::ilog << "BoxSize      " << header_.BoxSize <<  std::endl;
}

int simbelmyne_output_plugin::get_species_idx(const cosmo_species &s) const
{
  switch (s)
  {
  case cosmo_species::dm:
    return 1;
  case cosmo_species::baryon:
    return 0;
  case cosmo_species::neutrino:
    return 3;
  }
  return -1;
}

void simbelmyne_output_plugin::write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species)
{
    int sid = get_species_idx(s);

    assert(sid != -1);

    header_.npart[sid] = pc.get_local_num_particles();
    header_.npartTotal[sid] = pc.get_global_num_particles();

    if( pc.bhas_individual_masses_ )
      header_.mass[sid] = 0.0;
    else
      header_.mass[sid] = Omega_species * munit_ / pc.get_global_num_particles();

    HDFCreateGroup(this_fname_, std::string("PartType") + std::to_string(sid));

    //... write positions and velocities.....
    if (this->has_64bit_reals())
    {
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Coordinates"), pc.positions64_);
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Velocities"), pc.velocities64_);
    }
    else
    {
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Coordinates"), pc.positions32_);
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Velocities"), pc.velocities32_);
    }

    //... write ids.....
    if (this->has_64bit_ids())
      HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/ParticleIDs"), pc.ids64_);
    else
      HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/ParticleIDs"), pc.ids32_);

    //... write masses.....
    if( pc.bhas_individual_masses_ ){
      if (this->has_64bit_reals()){
        HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Masses"), pc.mass64_);
      }else{
        HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Masses"), pc.mass32_);
      }
    }
    music::ilog << interface_name_ << " : Wrote " << pc.get_local_num_particles() << " particles of type" << sid 
    << " in Gadget4 format to file \'" << this_fname_ << "\'" << std::endl;

    write_gadget_header();
}



// Grid functions
std::string simbelmyne_output_plugin::get_field_name( const cosmo_species &s, const fluid_component &c )
{
	std::string field_name;
	switch( s ){
		case cosmo_species::dm: 
			field_name += "DM"; break;
		case cosmo_species::baryon: 
			field_name += "BA"; break;
		case cosmo_species::neutrino: 
			field_name += "NU"; break;
		default: break;
	}
	field_name += "_";
	switch( c ){
		case fluid_component::density:
			field_name += "delta"; break;
		case fluid_component::vx:
			field_name += "vx"; break;
		case fluid_component::vy:
			field_name += "vy"; break;
		case fluid_component::vz:
			field_name += "vz"; break;
		case fluid_component::dx:
			field_name += "dx"; break;
		case fluid_component::dy:
			field_name += "dy"; break;
		case fluid_component::dz:
			field_name += "dz"; break;
        case fluid_component::mass:
            field_name += "mass"; break;
        case fluid_component::phi:
            field_name += "phi"; break;
        case fluid_component::phi2:
            field_name += "phi2"; break;
        case fluid_component::phi3:
            field_name += "phi3"; break;
        case fluid_component::A1:
            field_name += "A1"; break;
        case fluid_component::A2:
            field_name += "A2"; break;
        case fluid_component::A3:
            field_name += "A3"; break;
		default: break;
	}
	return field_name;
}


template< typename T > 
void simbelmyne_output_plugin::write_header_attribute( const std::string Filename, const std::string ObjName, const T &Data )
{
    hid_t dataspace_id, attribute_id, HDF_FileID, HDF_DatatypeID;
    HDF_DatatypeID = GetDataType<T>();  // Get the HDF5 datatype for the template type
    HDF_FileID = H5Fopen( Filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );  
    dataspace_id = H5Screate(H5S_SCALAR);
    attribute_id = H5Acreate2(HDF_FileID, ObjName.c_str(), HDF_DatatypeID, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute_id, HDF_DatatypeID, &Data);
    H5Aclose(attribute_id);
    H5Sclose(dataspace_id);
    H5Fclose( HDF_FileID ); 
}

void simbelmyne_output_plugin::add_simbelmyne_metadata( const std::string &fname )
{
    double L0 = cf_.get_value<double>("setup", "BoxLength");
    double L1=L0, L2=L0;
    double corner0=0.0, corner1=0.0, corner2=0.0;
    int N0 = cf_.get_value<int>("setup", "GridRes");
    int N1=N0, N2=N0;
    int rank = 1;
    double time = 1.0/(1.0+cf_.get_value<double>("setup", "zstart"));

    write_header_attribute<double>(fname, "/info/scalars/L0", L0);
    write_header_attribute<double>(fname, "/info/scalars/L1", L1);
    write_header_attribute<double>(fname, "/info/scalars/L2", L2);
    write_header_attribute<double>(fname, "/info/scalars/corner0", corner0);
    write_header_attribute<double>(fname, "/info/scalars/corner1", corner1);
    write_header_attribute<double>(fname, "/info/scalars/corner2", corner2);
    write_header_attribute<double>(fname, "/info/scalars/time", time);
    write_header_attribute<int>(fname, "/info/scalars/N0", N0);
    write_header_attribute<int>(fname, "/info/scalars/N1", N1);
    write_header_attribute<int>(fname, "/info/scalars/N2", N2);
    write_header_attribute<int>(fname, "/info/scalars/rank", rank);
}

void simbelmyne_output_plugin::move_dataset_in_hdf5( const std::string &fname, const std::string &src_dset_name, const std::string &group_name, const std::string &tg_dset_name )
{
    hid_t file_id, dataset_id, new_group_id;

    // Open the existing HDF5 file
    file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        music::elog << "Unable to open file " << fname << std::endl;
        return;
    }

    // Open the dataset provided by the user
    dataset_id = H5Dopen(file_id, src_dset_name.c_str());
    if (dataset_id < 0) {
        music::elog << "Unable to open dataset " << src_dset_name << std::endl;
        H5Fclose(file_id);
        return;
    }
    H5Dclose(dataset_id);

    // Create the target group "/scalars" if it doesn't exist
    if (!H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT)) {
        new_group_id = H5Gcreate(file_id, group_name.c_str(), H5P_DEFAULT);
        if (new_group_id < 0) {
            music::elog << "Unable to create goup " << group_name << std::endl;
            H5Fclose(file_id);
            return;
        }
        H5Gclose(new_group_id);
    }

    // Move the dataset to "/scalars/field"
    if (H5Lmove(file_id, src_dset_name.c_str(), file_id, (group_name+"/"+tg_dset_name).c_str(), H5P_DEFAULT, H5P_DEFAULT) < 0) {
        music::elog << "Unable to move dataset " << src_dset_name << " to " << (group_name+"/"+tg_dset_name) << std::endl;
        H5Fclose(file_id);
        return;
    }

    // Close dataset and file
    H5Fclose(file_id);
}



void simbelmyne_output_plugin::write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c ) 
{
    std::string field_name = "field";
    std::string file_subname = this->get_field_name( s, c );
    std::string file_name = fname_ + file_subname + ".h5";

    if( CONFIG::MPI_task_rank == 0 )
    {
        HDFCreateFile( file_name );
        // Simbelmyne header
        HDFCreateGroup( file_name, "info" );
        HDFCreateSubGroup( file_name, "info", "scalars" );
        add_simbelmyne_metadata(file_name);
    }

    #if defined(USE_MPI)
        MPI_Barrier( MPI_COMM_WORLD );
    #endif

    // Write the dataset
    g.Write_to_HDF5(file_name, field_name);

    #if defined(USE_MPI)
        MPI_Barrier( MPI_COMM_WORLD );
    #endif

    if( CONFIG::MPI_task_rank == 0 )
    {
        // Move dataset to "/scalars/field"
        HDFCreateGroup( file_name, "scalars" );
        move_dataset_in_hdf5(file_name, field_name, "scalars", "field");
    }


    music::ilog << interface_name_ << " : Wrote field \'" << field_name 
                << "\' with Simbelmyne metadata to file \'" << file_name << "\'" << std::endl;
}

namespace
{
   output_plugin_creator_concrete<simbelmyne_output_plugin> creator501("simbelmyne"); 
} // namespace
