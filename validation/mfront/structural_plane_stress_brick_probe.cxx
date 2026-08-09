/*
 * Minimal external Behaviour Brick registration probe.
 *
 * This is deliberately not a plane-stress implementation.  It only checks
 * that an external module can register a first-level brick in the installed
 * TFEL/MFront 5.1 parser through MFRONT_ADDITIONAL_LIBRARIES.
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MFront/BehaviourBrick/OptionDescription.hxx"
#include "MFront/BehaviourBrickBase.hxx"
#include "MFront/BehaviourBrickFactory.hxx"

namespace {

struct StructuralPlaneStress3DProbe final : mfront::BehaviourBrickBase {
  StructuralPlaneStress3DProbe(mfront::AbstractBehaviourDSL& dsl,
                               mfront::BehaviourDescription& behaviour)
      : mfront::BehaviourBrickBase(dsl, behaviour) {}

  std::string getName() const override { return "StructuralPlaneStress3DProbe"; }

  mfront::BehaviourBrickDescription getDescription() const override {
    mfront::BehaviourBrickDescription description;
    description.behaviourType =
        tfel::material::MechanicalBehaviourBase::GENERALBEHAVIOUR;
    description.integrationScheme = mfront::IntegrationScheme::IMPLICITSCHEME;
    description.managedCodeBlocks = {};
    return description;
  }

  std::vector<mfront::bbrick::OptionDescription> getOptions(
      const bool) const override {
    return {};
  }

  void initialize(const Parameters&, const DataMap&) override {}

  std::vector<Hypothesis> getSupportedModellingHypotheses() const override {
    return {};
  }

  void endTreatment() const override {}

  void completeVariableDeclaration() const override {}
};

std::shared_ptr<mfront::AbstractBehaviourBrick> makeProbe(
    mfront::AbstractBehaviourDSL& dsl, mfront::BehaviourDescription& behaviour) {
  return std::make_shared<StructuralPlaneStress3DProbe>(dsl, behaviour);
}

struct RegisterProbe final {
  RegisterProbe() {
    auto& factory = mfront::BehaviourBrickFactory::getFactory();
    factory.registerAbstractBehaviourBrick(
        "StructuralPlaneStress3DProbe",
        tfel::material::MechanicalBehaviourBase::GENERALBEHAVIOUR,
        mfront::IntegrationScheme::IMPLICITSCHEME, &makeProbe);
  }
};

const RegisterProbe register_probe{};

}  // namespace
