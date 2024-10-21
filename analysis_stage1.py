import os, sys
import urllib.request

processList = {
    'tt_incl_eec_365_IDEA': {
        "fraction": 1.0,
    }
}

outputDir = '.'

#inputDir = '/pnfs/iihe/cms/store/user/kskovpen/FCS/edm'
inputDir = '/user/kskovpen/analysis/FCS/key4hep/jobs_edm/tt_incl_eec_365_IDEA'

includePaths = ["functions.h"]

nCPUS = 4

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc_v1"

## model files needed for unit testing in CI
url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## model files locally stored on /eos
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

## get local file, else download from url
def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)

weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

output_branches = []

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:
    def analysers(df):

        df = df.Alias("Particle0", "_Particle_parents.index")
        df = df.Alias("Particle1", "_Particle_daughters.index")

        df = df.Define('GenTop1', 'MCParticle::sel_pdgID(6, false)(Particle)')
        df = df.Define('GenTop2', 'MCParticle::sel_pdgID(-6, false)(Particle)')
        
        df = df.Define("GenTop1_px",     "MCParticle::get_px(GenTop1)")
        df = df.Define("GenTop1_py",     "MCParticle::get_py(GenTop1)")
        df = df.Define("GenTop1_pz",     "MCParticle::get_pz(GenTop1)")
        df = df.Define("GenTop1_energy", "MCParticle::get_e(GenTop1)")
        df = df.Define("GenTop1_mass",   "MCParticle::get_mass(GenTop1)")
        df = df.Define("GenTop1_charge", "MCParticle::get_charge(GenTop1)")

        df = df.Define("GenTop2_px",     "MCParticle::get_px(GenTop2)")
        df = df.Define("GenTop2_py",     "MCParticle::get_py(GenTop2)")
        df = df.Define("GenTop2_pz",     "MCParticle::get_pz(GenTop2)")
        df = df.Define("GenTop2_energy", "MCParticle::get_e(GenTop2)")
        df = df.Define("GenTop2_mass",   "MCParticle::get_mass(GenTop2)")
        df = df.Define("GenTop2_charge", "MCParticle::get_charge(GenTop2)")
        
        # top decay
        df = df.Define("GenTop1_decay",
        "auto p = MCParticle::get_list_of_particles_from_decay(0, {GenTop1}, Particle1);\
        while(p.size() != 2) {\
          p = MCParticle::get_list_of_particles_from_decay(0, {MCParticle::sel_byIndex(p.at(0), Particle)}, Particle1);\
        }; return p;")
        df = df.Define("GenTop2_decay",
        "auto p = MCParticle::get_list_of_particles_from_decay(0, {GenTop2}, Particle1);\
        while(p.size() != 2) {\
          p = MCParticle::get_list_of_particles_from_decay(0, {MCParticle::sel_byIndex(p.at(0), Particle)}, Particle1);\
        }; return p;")
        df = df.Define("GenTop1P1", "MCParticle::sel_byIndex(GenTop1_decay.at(0), Particle)")
        df = df.Define("GenTop1P2", "MCParticle::sel_byIndex(GenTop1_decay.at(1), Particle)")
        df = df.Define("GenTop2P1", "MCParticle::sel_byIndex(GenTop2_decay.at(0), Particle)")
        df = df.Define("GenTop2P2", "MCParticle::sel_byIndex(GenTop2_decay.at(1), Particle)")
        df = df.Define("GenTop1W", "(abs(GenTop1P1.PDG) == 24) ? GenTop1P1 : GenTop1P2")
        df = df.Define("GenTop1B", "(abs(GenTop1P1.PDG) == 5) ? GenTop1P1 : GenTop1P2")
        df = df.Define("GenTop2W", "(abs(GenTop2P1.PDG) == 24) ? GenTop2P1 : GenTop2P2")
        df = df.Define("GenTop2B", "(abs(GenTop2P1.PDG) == 5) ? GenTop2P1 : GenTop2P2")

#        df = df.Define("Dummy", "ROOT::VecOps::RVec<float>{-1}")
        
        df = df.Define("GenTop1W_px",       "MCParticle::get_px({GenTop1W})")
        df = df.Define("GenTop1W_py",       "MCParticle::get_py({GenTop1W})")
        df = df.Define("GenTop1W_pz",       "MCParticle::get_pz({GenTop1W})")
        df = df.Define("GenTop1W_energy",   "MCParticle::get_e({GenTop1W})")
        df = df.Define("GenTop1W_mass",     "MCParticle::get_mass({GenTop1W})")
        df = df.Define("GenTop1W_charge",   "MCParticle::get_charge({GenTop1W})")
        df = df.Define("GenTop1W_pdg",      "MCParticle::get_pdg({GenTop1W})")

        df = df.Define("GenTop1B_px",       "MCParticle::get_px({GenTop1B})")
        df = df.Define("GenTop1B_py",       "MCParticle::get_py({GenTop1B})")
        df = df.Define("GenTop1B_pz",       "MCParticle::get_pz({GenTop1B})")
        df = df.Define("GenTop1B_energy",   "MCParticle::get_e({GenTop1B})")
        df = df.Define("GenTop1B_mass",     "MCParticle::get_mass({GenTop1B})")
        df = df.Define("GenTop1B_charge",   "MCParticle::get_charge({GenTop1B})")
        df = df.Define("GenTop1B_pdg",      "MCParticle::get_pdg({GenTop1B})")

        df = df.Define("GenTop2W_px",       "MCParticle::get_px({GenTop2W})")
        df = df.Define("GenTop2W_py",       "MCParticle::get_py({GenTop2W})")
        df = df.Define("GenTop2W_pz",       "MCParticle::get_pz({GenTop2W})")
        df = df.Define("GenTop2W_energy",   "MCParticle::get_e({GenTop2W})")
        df = df.Define("GenTop2W_mass",     "MCParticle::get_mass({GenTop2W})")
        df = df.Define("GenTop2W_charge",   "MCParticle::get_charge({GenTop2W})")
        df = df.Define("GenTop2W_pdg",      "MCParticle::get_pdg({GenTop2W})")

        df = df.Define("GenTop2B_px",       "MCParticle::get_px({GenTop2B})")
        df = df.Define("GenTop2B_py",       "MCParticle::get_py({GenTop2B})")
        df = df.Define("GenTop2B_pz",       "MCParticle::get_pz({GenTop2B})")
        df = df.Define("GenTop2B_energy",   "MCParticle::get_e({GenTop2B})")
        df = df.Define("GenTop2B_mass",     "MCParticle::get_mass({GenTop2B})")
        df = df.Define("GenTop2B_charge",   "MCParticle::get_charge({GenTop2B})")
        df = df.Define("GenTop2B_pdg",      "MCParticle::get_pdg({GenTop2B})")
        
        # w decay
        df = df.Define("GenTop1W_decay",
        "auto p = MCParticle::get_list_of_particles_from_decay(0, {GenTop1W}, Particle1);\
        while(p.size() != 2) {\
          p = MCParticle::get_list_of_particles_from_decay(0, {MCParticle::sel_byIndex(p.at(0), Particle)}, Particle1);\
        }; return p;")
        df = df.Define("GenTop2W_decay",
        "auto p = MCParticle::get_list_of_particles_from_decay(0, {GenTop2W}, Particle1);\
        while(p.size() != 2) {\
          p = MCParticle::get_list_of_particles_from_decay(0, {MCParticle::sel_byIndex(p.at(0), Particle)}, Particle1);\
        }; return p;")
        df = df.Define("GenTop1WP1", "MCParticle::sel_byIndex(GenTop1W_decay.at(0), Particle)")
        df = df.Define("GenTop1WP2", "MCParticle::sel_byIndex(GenTop1W_decay.at(1), Particle)")
        df = df.Define("GenTop2WP1", "MCParticle::sel_byIndex(GenTop2W_decay.at(0), Particle)")
        df = df.Define("GenTop2WP2", "MCParticle::sel_byIndex(GenTop2W_decay.at(1), Particle)")

        df = df.Define("GenTop1WP1_px",       "MCParticle::get_px({GenTop1WP1})")
        df = df.Define("GenTop1WP1_py",       "MCParticle::get_py({GenTop1WP1})")
        df = df.Define("GenTop1WP1_pz",       "MCParticle::get_pz({GenTop1WP1})")
        df = df.Define("GenTop1WP1_energy",   "MCParticle::get_e({GenTop1WP1})")
        df = df.Define("GenTop1WP1_mass",     "MCParticle::get_mass({GenTop1WP1})")
        df = df.Define("GenTop1WP1_charge",   "MCParticle::get_charge({GenTop1WP1})")
        df = df.Define("GenTop1WP1_pdg",      "MCParticle::get_pdg({GenTop1WP1})")

        df = df.Define("GenTop1WP2_px",       "MCParticle::get_px({GenTop1WP2})")
        df = df.Define("GenTop1WP2_py",       "MCParticle::get_py({GenTop1WP2})")
        df = df.Define("GenTop1WP2_pz",       "MCParticle::get_pz({GenTop1WP2})")
        df = df.Define("GenTop1WP2_energy",   "MCParticle::get_e({GenTop1WP2})")
        df = df.Define("GenTop1WP2_mass",     "MCParticle::get_mass({GenTop1WP2})")
        df = df.Define("GenTop1WP2_charge",   "MCParticle::get_charge({GenTop1WP2})")
        df = df.Define("GenTop1WP2_pdg",      "MCParticle::get_pdg({GenTop1WP2})")
        
        df = df.Define("GenTop2WP1_px",       "MCParticle::get_px({GenTop2WP1})")
        df = df.Define("GenTop2WP1_py",       "MCParticle::get_py({GenTop2WP1})")
        df = df.Define("GenTop2WP1_pz",       "MCParticle::get_pz({GenTop2WP1})")
        df = df.Define("GenTop2WP1_energy",   "MCParticle::get_e({GenTop2WP1})")
        df = df.Define("GenTop2WP1_mass",     "MCParticle::get_mass({GenTop2WP1})")
        df = df.Define("GenTop2WP1_charge",   "MCParticle::get_charge({GenTop2WP1})")
        df = df.Define("GenTop2WP1_pdg",      "MCParticle::get_pdg({GenTop2WP1})")

        df = df.Define("GenTop2WP2_px",       "MCParticle::get_px({GenTop2WP2})")
        df = df.Define("GenTop2WP2_py",       "MCParticle::get_py({GenTop2WP2})")
        df = df.Define("GenTop2WP2_pz",       "MCParticle::get_pz({GenTop2WP2})")
        df = df.Define("GenTop2WP2_energy",   "MCParticle::get_e({GenTop2WP2})")
        df = df.Define("GenTop2WP2_mass",     "MCParticle::get_mass({GenTop2WP2})")
        df = df.Define("GenTop2WP2_charge",   "MCParticle::get_charge({GenTop2WP2})")
        df = df.Define("GenTop2WP2_pdg",      "MCParticle::get_pdg({GenTop2WP2})")
        
        df = df.Alias("Muon0", "Muon_objIdx.index")
        df = df.Alias("Photon0", "Photon_objIdx.index")
        df = df.Alias("Electron0", "Electron_objIdx.index")

        df = df.Define("RP_px",          "ReconstructedParticle::get_px(ReconstructedParticles)")
        df = df.Define("RP_py",          "ReconstructedParticle::get_py(ReconstructedParticles)")
        df = df.Define("RP_pz",          "ReconstructedParticle::get_pz(ReconstructedParticles)")
        df = df.Define("RP_e",           "ReconstructedParticle::get_e(ReconstructedParticles)")
        df = df.Define("RP_m",           "ReconstructedParticle::get_mass(ReconstructedParticles)")
        df = df.Define("RP_q",           "ReconstructedParticle::get_charge(ReconstructedParticles)")

        df = df.Define("muons_all", "ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("muons", "ReconstructedParticle::sel_pt(3)(muons_all)")

        df = df.Define("muons_iso", "FCCAnalyses::ANAfunctions::coneIsolation(0.01, 0.5)(muons, ReconstructedParticles)")
        df = df.Define("muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons)")
        df = df.Define("muons_e", "FCCAnalyses::ReconstructedParticle::get_e(muons)")
        df = df.Define("muons_px", "FCCAnalyses::ReconstructedParticle::get_px(muons)")
        df = df.Define("muons_py", "FCCAnalyses::ReconstructedParticle::get_py(muons)")
        df = df.Define("muons_pz", "FCCAnalyses::ReconstructedParticle::get_pz(muons)")
        df = df.Define("muons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(muons)")
        df = df.Define("muons_eta", "FCCAnalyses::ReconstructedParticle::get_eta(muons)")
        df = df.Define("muons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(muons)")
        df = df.Define("muons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons)")
        df = df.Define("muons_q", "FCCAnalyses::ReconstructedParticle::get_charge(muons)")
        df = df.Define("muons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons)")

        df = df.Define("electrons_all", "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("electrons", "ReconstructedParticle::sel_pt(3)(electrons_all)")

        df = df.Define("electrons_iso", "FCCAnalyses::ANAfunctions::coneIsolation(0.01, 0.5)(electrons, ReconstructedParticles)")
        df = df.Define("electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons)")
        df = df.Define("electrons_e", "FCCAnalyses::ReconstructedParticle::get_e(electrons)")
        df = df.Define("electrons_px", "FCCAnalyses::ReconstructedParticle::get_px(electrons)")
        df = df.Define("electrons_py", "FCCAnalyses::ReconstructedParticle::get_py(electrons)")
        df = df.Define("electrons_pz", "FCCAnalyses::ReconstructedParticle::get_pz(electrons)")
        df = df.Define("electrons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(electrons)")
        df = df.Define("electrons_eta", "FCCAnalyses::ReconstructedParticle::get_eta(electrons)")
        df = df.Define("electrons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(electrons)")
        df = df.Define("electrons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons)")
        df = df.Define("electrons_q", "FCCAnalyses::ReconstructedParticle::get_charge(electrons)")
        df = df.Define("electrons_n", "FCCAnalyses::ReconstructedParticle::get_n(electrons)")

        df = df.Define("photons_all", "ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df = df.Define("photons", "ReconstructedParticle::sel_pt(3)(photons_all)")

        df = df.Define("photons_iso", "FCCAnalyses::ANAfunctions::coneIsolation(0.01, 0.5)(photons, ReconstructedParticles)")
        df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons)")
        df = df.Define("photons_e", "FCCAnalyses::ReconstructedParticle::get_e(photons)")
        df = df.Define("photons_px", "FCCAnalyses::ReconstructedParticle::get_px(photons)")
        df = df.Define("photons_py", "FCCAnalyses::ReconstructedParticle::get_py(photons)")
        df = df.Define("photons_pz", "FCCAnalyses::ReconstructedParticle::get_pz(photons)")
        df = df.Define("photons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(photons)")
        df = df.Define("photons_eta", "FCCAnalyses::ReconstructedParticle::get_eta(photons)")
        df = df.Define("photons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(photons)")
        df = df.Define("photons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(photons)")
        df = df.Define("photons_n", "FCCAnalyses::ReconstructedParticle::get_n(photons)")

        df = df.Redefine("MissingET", "ANAfunctions::missingEnergy(365, ReconstructedParticles)")

        df = df.Define("MET_e", "ReconstructedParticle::get_e(MissingET)")
        df = df.Define("MET_p", "ReconstructedParticle::get_p(MissingET)")
        df = df.Define("MET_pt", "ReconstructedParticle::get_pt(MissingET)")
        df = df.Define("MET_px", "ReconstructedParticle::get_px(MissingET)")
        df = df.Define("MET_py", "ReconstructedParticle::get_py(MissingET)")
        df = df.Define("MET_pz", "ReconstructedParticle::get_pz(MissingET)")
        df = df.Define("MET_eta", "ReconstructedParticle::get_eta(MissingET)")
        df = df.Define("MET_theta", "ReconstructedParticle::get_theta(MissingET)")
        df = df.Define("MET_phi", "ReconstructedParticle::get_phi(MissingET)")
        
        df = df.Define('electrons_tracks', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK(electrons, _EFlowTrack_trackStates)')
        df = df.Define('electrons_d0', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_D0(electrons, _EFlowTrack_trackStates)')
        df = df.Define('electrons_d0signif', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_D0_sig(electrons, _EFlowTrack_trackStates)')
        df = df.Define('electrons_z0', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_Z0(electrons, _EFlowTrack_trackStates)')
        df = df.Define('electrons_z0signif', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_Z0_sig(electrons, _EFlowTrack_trackStates)')
        
        df = df.Define('muons_tracks', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK(muons, _EFlowTrack_trackStates)')
        df = df.Define('muons_d0', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_D0(muons, _EFlowTrack_trackStates)')
        df = df.Define('muons_d0signif', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_D0_sig(muons, _EFlowTrack_trackStates)')
        df = df.Define('muons_z0', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_Z0(muons, _EFlowTrack_trackStates)')
        df = df.Define('muons_z0signif', 'FCCAnalyses::ReconstructedParticle2Track::getRP2TRK_Z0_sig(muons, _EFlowTrack_trackStates)')
        
        #        df = df.Define("VertexObject_DiElectrons",  "FCCAnalyses::VertexFitterSimple::VertexFitter_Tk( 2, electrons_tracks)" )
        #        df = df.Define("Vertex_DiElectrons",   "FCCAnalyses::VertexingUtils::get_VertexData( VertexObject_DiElectrons )")
        
##        df = df.Define("VertexObject_tracks",  "FCCAnalyses::VertexFitterSimple::VertexFitter_Tk( 2, _EFlowTrack_trackStates)" )
##        df = df.Define("Vertex",   "FCCAnalyses::VertexingUtils::get_VertexData( VertexObject_tracks )")
##        df = df.Define("vtx_x",   "Vertex.position.x")
##        df = df.Define("vtx_y",   "Vertex.position.y")
##        df = df.Define("vtx_z",   "Vertex.position.z")
##        df = df.Define("vtx_chi2",   "Vertex.chi2")

#        df = df.Alias("ReconstructedTracksStates", "_SiTracks_Refitted_trackStates")
#        df = df.Alias("ReconstructedTracksData", "SiTracks")
#        df = df.Define("TracksAtIP", "FCCAnalyses::ReconstructedTrack::TrackStates_at_IP(ReconstructedTracksData, ReconstructedTracksStates)")
#        df = df.Define("trackStates_electrons", "FCCAnalyses::ReconstructedParticle2Track::getRP2TRK(electrons, TracksAtIP)")
#        df = df.Define("VertexObject_DiElectrons",  "FCCAnalyses::VertexFitterSimple::VertexFitter ( 1, electrons, trackStates_electrons) ")

        df = df.Define(
            "muons_sel_iso",
            "FCCAnalyses::ANAfunctions::sel_iso(0.25)(muons, muons_iso)",
        )
        
        df = df.Define(
            "electrons_sel_iso",
            "FCCAnalyses::ANAfunctions::sel_iso(0.25)(electrons, electrons_iso)",
        )

        df = df.Define(
            "ReconstructedParticlesNoIsoMuons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles,muons_sel_iso)",
        )

        df = df.Define(
            "ReconstructedParticlesNoIsoLeptons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoIsoMuons,electrons_sel_iso)",
        )

        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticlesNoIsoLeptons",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "_EFlowTrack_trackStates",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "_EFlowTrack_dxQuantities",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

#        jetClusteringHelper = ExclusiveJetClusteringHelper(collections["PFParticles"], 4, "ee_kt")
#        df = jetClusteringHelper.define(df)

#        jetFlavourHelper = JetFlavourHelper(collections, jetClusteringHelper.jets, jetClusteringHelper.constituents, "ee_kt")
#        df = jetFlavourHelper.define(df)
#        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

#        df = df.Define('event_thrust',     'Algorithms::minimize_thrust("Minuit2","Migrad")(RP_px, RP_py, RP_pz)')
#        df = df.Define('RP_thrustangle',   'Algorithms::getAxisCosTheta(event_thrust, RP_px, RP_py, RP_pz)')
#        df = df.Define('event_thrust_x',   "event_thrust.at(0)")
#        df = df.Define('event_thrust_y',   "event_thrust.at(1)")
#        df = df.Define('event_thrust_z',   "event_thrust.at(2)")
#        df = df.Define('event_thrust_val', "event_thrust.at(3)")

#        df = df.Define('event_sphericity',     'Algorithms::minimize_sphericity("Minuit2","Migrad")(RP_px, RP_py, RP_pz)')
#        df = df.Define('event_sphericity_x',   "event_sphericity.at(0)")
#        df = df.Define('event_sphericity_y',   "event_sphericity.at(1)")
#        df = df.Define('event_sphericity_z',   "event_sphericity.at(2)")
#        df = df.Define('event_sphericity_val', "event_sphericity.at(3)")
#        df = df.Define('RP_sphericityangle', 'Algorithms::getAxisCosTheta(event_sphericity, RP_px, RP_py, RP_pz)')

#        df = df.Define('RP_hemis0_mass',   "Algorithms::getAxisMass(0)(RP_thrustangle, RP_e, RP_px, RP_py, RP_pz)")
#        df = df.Define('RP_hemis1_mass',   "Algorithms::getAxisMass(1)(RP_thrustangle, RP_e, RP_px, RP_py, RP_pz)")

#        df = df.Define("RP_total_mass",    "Algorithms::getMass(ReconstructedParticles)")

#        df = df.Define("jet_ee_kt_dmerge1", "JetClusteringUtils::get_exclusive_dmerge(_jet_ee_kt, 1)")
#        df = df.Define("jet_ee_kt_dmerge2", "JetClusteringUtils::get_exclusive_dmerge(_jet_ee_kt, 2)")
#        df = df.Define("jet_ee_kt_dmerge3", "JetClusteringUtils::get_exclusive_dmerge(_jet_ee_kt, 3)")

#        df = df.Define("jet_ee_kt_px",        "JetClusteringUtils::get_px(jet_ee_kt)")
#        df = df.Define("jet_ee_kt_py",        "JetClusteringUtils::get_py(jet_ee_kt)")
#        df = df.Define("jet_ee_kt_pz",        "JetClusteringUtils::get_pz(jet_ee_kt)")
#        df = df.Define("jet_ee_kt_pt",        "JetClusteringUtils::get_pt(jet_ee_kt)")
#        df = df.Define("jet_ee_kt_eta",       "JetClusteringUtils::get_eta(jet_ee_kt)")
#        df = df.Define("jet_ee_kt_phi_std",   "JetClusteringUtils::get_phi_std(jet_ee_kt)")
#        df = df.Alias("jet_ee_kt_n",          "event_njet_ee_kt")

        global output_branches
        output_branches = set()
        
#        for i in set([*jetClusteringHelper.outputBranches(), *jetFlavourHelper.outputBranches()]):
#            if i.startswith("jet_"): b = i[4:i.find("_ee_kt")]
#            elif i.startswith("recojet_"): b = i[8:i.find("_ee_kt")]
#            else: continue
#            df = df.Alias(f"jet_ee_kt_{b}", i)
#            output_branches.add(f"jet_ee_kt_{b}")
            
        df = df.Define("RPnl_px",          "ReconstructedParticle::get_px(ReconstructedParticlesNoIsoLeptons)")
        df = df.Define("RPnl_py",          "ReconstructedParticle::get_py(ReconstructedParticlesNoIsoLeptons)")
        df = df.Define("RPnl_pz",          "ReconstructedParticle::get_pz(ReconstructedParticlesNoIsoLeptons)")
        df = df.Define("RPnl_e",           "ReconstructedParticle::get_e(ReconstructedParticlesNoIsoLeptons)")
        df = df.Define("RPnl_m",           "ReconstructedParticle::get_mass(ReconstructedParticlesNoIsoLeptons)")
        df = df.Define("RPnl_q",           "ReconstructedParticle::get_charge(ReconstructedParticlesNoIsoLeptons)")
        
        df = df.Define("pseudo_jets",            "JetClusteringUtils::set_pseudoJets_xyzm(RPnl_px, RPnl_py, RPnl_pz, RPnl_m)")
        df = df.Define("FCCAnalysesJets_antikt", "JetClustering::clustering_antikt(0.5, 0, 20, 0, 0)(pseudo_jets)")
        df = df.Define("jet_antikt",             "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_antikt)")
        df = df.Define("jetconstituents_antikt", "JetClusteringUtils::get_constituents(FCCAnalysesJets_antikt)")
        df = df.Define("jetc_antikt",            "JetConstituentsUtils::build_constituents_cluster(ReconstructedParticlesNoIsoLeptons,jetconstituents_antikt)")
        df = df.Define("jet_antikt_nconst",      "JetConstituentsUtils::count_consts(jetc_antikt)")
        df = df.Define("jet_antikt_e",           "JetClusteringUtils::get_e(jet_antikt)")
        df = df.Define("jet_antikt_px",          "JetClusteringUtils::get_px(jet_antikt)")
        df = df.Define("jet_antikt_py",          "JetClusteringUtils::get_py(jet_antikt)")
        df = df.Define("jet_antikt_pz",          "JetClusteringUtils::get_pz(jet_antikt)")
        df = df.Define("jet_antikt_pt",          "JetClusteringUtils::get_pt(jet_antikt)")
        df = df.Define("jet_antikt_p",           "JetClusteringUtils::get_p(jet_antikt)")
        df = df.Define("jet_antikt_phi",         "JetClusteringUtils::get_phi(jet_antikt)")
        df = df.Define("jet_antikt_eta",         "JetClusteringUtils::get_eta(jet_antikt)")
        df = df.Define("jet_antikt_mass",        "JetClusteringUtils::get_m(jet_antikt)")
        df = df.Define("jet_antikt_n",           "JetConstituentsUtils::count_jets(jetc_antikt)")
        
        jetFlavourHelper = JetFlavourHelper(collections, "jet_antikt", "jetc_antikt", "antikt")
        df = jetFlavourHelper.define(df)
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)
        
        for i in set(jetFlavourHelper.outputBranches()):
            if i.startswith("jet_"): b = i[4:i.find("_antikt")]
            elif i.startswith("recojet_"): b = i[8:i.find("_antikt")]
            else: continue
            df = df.Alias(f"jet_antikt_{b}", i)
            output_branches.add(f"jet_antikt_{b}")
        
        return df
    
    def output():
        branchList = [
#            "RP_px", "RP_py", "RP_pz", "RP_e", "RP_m", "RP_q",

#            "event_thrust_x", "event_thrust_y", "event_thrust_z", "event_thrust_val",

#            "event_sphericity_x", "event_sphericity_y", "event_sphericity_z", "event_sphericity_val",

#            "RP_thrustangle",
#            "RP_sphericityangle",

#            "RP_hemis0_mass",
#            "RP_hemis1_mass",
#            "RP_total_mass",

            "GenTop1_px",
            "GenTop1_py",
            "GenTop1_pz",
            "GenTop1_energy",
            "GenTop1_mass",
            "GenTop1_charge",

            "GenTop2_px",
            "GenTop2_py",
            "GenTop2_pz",
            "GenTop2_energy",
            "GenTop2_mass",
            "GenTop2_charge",

            "GenTop1W_px",
            "GenTop1W_py",
            "GenTop1W_pz",
            "GenTop1W_energy",
            "GenTop1W_mass",
            "GenTop1W_charge",
            "GenTop1W_pdg",

            "GenTop1B_px",
            "GenTop1B_py",
            "GenTop1B_pz",
            "GenTop1B_energy",
            "GenTop1B_mass",
            "GenTop1B_charge",
            "GenTop1B_pdg",

            "GenTop2W_px",
            "GenTop2W_py",
            "GenTop2W_pz",
            "GenTop2W_energy",
            "GenTop2W_mass",
            "GenTop2W_charge",
            "GenTop2W_pdg",

            "GenTop2B_px",
            "GenTop2B_py",
            "GenTop2B_pz",
            "GenTop2B_energy",
            "GenTop2B_mass",
            "GenTop2B_charge",
            "GenTop2B_pdg", 
           
            "GenTop1WP1_px",
            "GenTop1WP1_py",
            "GenTop1WP1_pz",
            "GenTop1WP1_energy",
            "GenTop1WP1_mass",
            "GenTop1WP1_charge",
            "GenTop1WP1_pdg",

            "GenTop1WP2_px",
            "GenTop1WP2_py",
            "GenTop1WP2_pz",
            "GenTop1WP2_energy",
            "GenTop1WP2_mass",
            "GenTop1WP2_charge",
            "GenTop1WP2_pdg",

            "GenTop2WP1_px",
            "GenTop2WP1_py",
            "GenTop2WP1_pz",
            "GenTop2WP1_energy",
            "GenTop2WP1_mass",
            "GenTop2WP1_charge",
            "GenTop2WP1_pdg",

            "GenTop2WP2_px",
            "GenTop2WP2_py",
            "GenTop2WP2_pz",
            "GenTop2WP2_energy",
            "GenTop2WP2_mass",
            "GenTop2WP2_charge",
            "GenTop2WP2_pdg",
            
            "muons_p",
            "muons_e",
            "muons_pt",
            "muons_px",
            "muons_py",
            "muons_pz",
            "muons_eta",
            "muons_theta",
            "muons_phi",
            "muons_q",
            "muons_iso",
            "muons_d0",
            "muons_d0signif",
            "muons_z0",
            "muons_z0signif",
            "muons_n",

            "electrons_p",
            "electrons_e",
            "electrons_pt",
            "electrons_px",
            "electrons_py",
            "electrons_pz",
            "electrons_eta",
            "electrons_theta",
            "electrons_phi",
            "electrons_q",
            "electrons_iso",
            "electrons_d0",
            "electrons_d0signif",
            "electrons_z0",
            "electrons_z0signif",
            "electrons_n",

            "photons_p",
            "photons_e",
            "photons_pt",
            "photons_px",
            "photons_py",
            "photons_pz",
            "photons_eta",
            "photons_theta",
            "photons_phi",
            "photons_iso",
            "photons_n",
            
            "MET_e",
            "MET_p",
            "MET_pt",
            "MET_px",
            "MET_py",
            "MET_pz",
            "MET_eta",
            "MET_theta",
            "MET_phi",
            
            "jet_antikt_e",    
            "jet_antikt_px",    
            "jet_antikt_py",    
            "jet_antikt_pz",    
            "jet_antikt_p",    
            "jet_antikt_pt",    
            "jet_antikt_phi",    
            "jet_antikt_eta",    
            "jet_antikt_mass",
            "jet_antikt_nconst",
            "jet_antikt_n"
            
##            "vtx_x",
##            "vtx_y",
##            "vtx_z",
##            "vtx_chi2"
            
#            "jet_ee_kt_px",
#            "jet_ee_kt_py",
#            "jet_ee_kt_pz",
#            "jet_ee_kt_pt",
#            "jet_ee_kt_phi",
#            "jet_ee_kt_eta",
#            "jet_ee_kt_e",
#            "jet_ee_kt_n"
        ]

        branchList += output_branches
        
        return branchList
