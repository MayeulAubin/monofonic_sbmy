#include <unistd.h> // for unlink

#include "HDF_IO.hh"
#include <logger.hh>
#include <output_plugin.hh>

class simbelmyne_output_plugin : public output_plugin
{
private:
    std::string get_field_name( const cosmo_species &s, const fluid_component &c );
    void add_simbelmyne_metadata( const std::string &fname );
    void move_dataset_in_hdf5( const std::string &fname, const std::string &dset_name, const std::string &group_name );

protected:
    bool out_eulerian_;

public:
    //! constructor
    explicit simbelmyne_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc )
    : output_plugin(cf, pcc, "Simbelmyne HDF5")
    {
        // out_eulerian_   = cf_.get_value_safe<bool>("output", "simbelmyne_out_eulerian", false);
        out_eulerian_ = true;
    }

    output_type write_species_as( const cosmo_species &s ) const
    { 
        if( out_eulerian_ )
            return output_type::field_eulerian;
        return output_type::field_lagrangian;
    }

    bool has_64bit_reals() const{ return true; }

    bool has_64bit_ids() const{ return true; }

    real_t position_unit() const { return 1.0; }
    
    real_t velocity_unit() const { return 1.0; }

    real_t mass_unit() const { return 1.0; }

    void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c );
};

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

void simbelmyne_output_plugin::add_simbelmyne_metadata( const std::string &fname )
{
    double L0 = cf_.get_value<double>("setup", "BoxLength");
    double L1=L0, L2=L0;
    double corner0=0.0, corner1=0.0, corner2=0.0;
    int N0 = cf_.get_value<int>("setup", "GridRes");
    int N1=N0, N2=N0;
    int rank = 1;

    HDFWriteGroupAttribute<double>(fname, "info/scalars", "L0", L0);
    HDFWriteGroupAttribute<double>(fname, "info/scalars", "L1", L1);
    HDFWriteGroupAttribute<double>(fname, "info/scalars", "L2", L2);
    HDFWriteGroupAttribute<double>(fname, "info/scalars", "corner0", corner0);
    HDFWriteGroupAttribute<double>(fname, "info/scalars", "corner1", corner1);
    HDFWriteGroupAttribute<double>(fname, "info/scalars", "corner2", corner2);
    HDFWriteGroupAttribute<int>(fname, "info/scalars", "N0", N0);
    HDFWriteGroupAttribute<int>(fname, "info/scalars", "N1", N1);
    HDFWriteGroupAttribute<int>(fname, "info/scalars", "N2", N2);
    HDFWriteGroupAttribute<int>(fname, "info/scalars", "rank", rank);
}

void simbelmyne_output_plugin::move_dataset_in_hdf5( const std::string &fname, const std::string &dset_name, const std::string &group_name )
{
    hid_t file_id, dataset_id, new_group_id;

    // Open the existing HDF5 file
    file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: Unable to open file %s\n", fname.c_str());
        return;
    }

    // Open the dataset provided by the user
    dataset_id = H5Dopen(file_id, dset_name.c_str());
    if (dataset_id < 0) {
        fprintf(stderr, "Error: Unable to open dataset %s\n", dset_name.c_str());
        H5Fclose(file_id);
        return;
    }

    // Create the target group "/scalars" if it doesn't exist
    if (!H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT)) {
        new_group_id = H5Gcreate(file_id, group_name.c_str(), H5P_DEFAULT);
        if (new_group_id < 0) {
            fprintf(stderr, "Error: Unable to create group %s\n", group_name.c_str());
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return;
        }
        H5Gclose(new_group_id);
    }

    // Move the dataset to "/scalars/field"
    if (H5Lmove(file_id, dset_name.c_str(), file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT) < 0) {
        fprintf(stderr, "Error: Unable to move dataset %s to %s\n", dset_name.c_str(), group_name.c_str());
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Close dataset and file
    H5Dclose(dataset_id);
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
        HDFCreateGroup( fname_, "info/scalars" );
        add_simbelmyne_metadata(fname_);
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
        move_dataset_in_hdf5(file_name, field_name, "/scalars/field");
    }


    music::ilog << interface_name_ << " : Wrote field \'" << field_name 
                << "\' with Simbelmyne metadata to file \'" << file_name << "\'" << std::endl;
}

namespace
{
   output_plugin_creator_concrete<simbelmyne_output_plugin> creator501("simbelmyne"); 
} // namespace
