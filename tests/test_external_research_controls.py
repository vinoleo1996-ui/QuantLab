import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from quantlab.external_research_controls import (
    D2ArchiveGate,
    ExternalResearchContractError,
    FixedSurface,
    MainlineMatrix,
    MiroFishRunContract,
    TextConsensus,
    ThemeCarrierEntryArtifact,
    entry_universe_contract_sha256,
    entry_only_instrument_set_sha256,
    validate_entry_sort_ab,
    validate_fusion_or_structure,
    validate_multi_agent_brief,
    validate_research_disclosure,
    register_atomic_entry_sort_experiment,
    u0_entry_universe_contract_sha256,
)
from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    U1,
    UniverseSnapshot,
    forward_membership_sha256,
)
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.theme_carrier import EntryOrdering, ThemeCarrierResult


class ExternalResearchControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._ledger_now = '2026-09-30 17:10:00+08:00'
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / 'research-signal-ledger', now_fn=lambda: self._ledger_now
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def digest(value):
        return hashlib.sha256(str(value).encode()).hexdigest()

    def decision_gate(self, *, status='archived_decided', selection_basis='pre_registered_full_matrix'):
        protocol_sha = self.digest('d2-protocol')
        archive_path = Path(self.temp.name) / f'd2-{status}-{selection_basis}.json'
        archive_identity_payload = dict(
            schema_version='quantlab.d2.full-matrix-archive.v1',
            run_id='full-50x5',
            protocol_sha256=protocol_sha,
            selection_basis=selection_basis,
            full_matrix_sha256=self.digest('d2-full-matrix'),
        )
        archive = dict(
            **archive_identity_payload,
            archive_identity_sha256=hashlib.sha256(
                json.dumps(archive_identity_payload, sort_keys=True, separators=(',', ':')).encode()
            ).hexdigest(),
        )
        archive_bytes = json.dumps(archive, sort_keys=True, separators=(',', ':')).encode()
        archive_path.write_bytes(archive_bytes)
        manifest = dict(
            schema_version='quantlab.d2.decision.v1',
            run_id='full-50x5',
            status=status,
            protocol_sha256=protocol_sha,
            selection_basis=selection_basis,
            archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
            archive_schema_version=archive['schema_version'],
            archive_identity_sha256=archive['archive_identity_sha256'],
        )
        decision_sha = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        return D2ArchiveGate('full-50x5', status, protocol_sha, decision_sha, manifest, str(archive_path))

    def theme_artifact(self, *, pool_id='U0'):
        spec = U0 if pool_id == 'U0' else U1
        rank_start = 3000 if pool_id == 'U0' else 5000
        members = tuple(f'sz.{rank_start + index:06d}' for index in range(1, 6))
        session = date(2026, 10, 1)
        decision_at = pd.Timestamp('2026-10-01 18:00:00+08:00')
        input_hash = self.digest(f'{pool_id}-forward-input')
        membership_hash = forward_membership_sha256(
            spec=spec,
            session=session,
            decision_at=decision_at,
            cadence_anchor=session,
            input_hash=input_hash,
            members=members,
            status='ok',
            abstain_reason=None,
        )
        universe = UniverseSnapshot(
            contract_version=FORWARD_CONTRACT_VERSION,
            pool_id=pool_id,
            session=session,
            decision_at=decision_at,
            cadence_anchor=session,
            members=members,
            rank_by_instrument={code: rank_start + index for index, code in enumerate(members, start=1)},
            adv60_by_instrument={code: 5_000_000.0 for code in members},
            eligibility_reason_by_instrument={},
            input_hash=input_hash,
            membership_hash=membership_hash,
            status='ok',
            abstain_reason=None,
        )
        components = pd.DataFrame(
            [
                dict(
                    theme_id='theme-a',
                    prior_limit_up_premium=.1,
                    broken_board_rate=.1,
                    theme_relative_strength=.1,
                    advancing_breadth=.5,
                    prior_limit_up_premium_percentile=.5,
                    broken_board_rate_percentile=.5,
                    theme_relative_strength_percentile=.5,
                    advancing_breadth_percentile=.5,
                    theme_score_equal_weight_v1=.5,
                    diagnostic_only_components=True,
                )
            ]
        )
        result = ThemeCarrierResult(
            status='ok',
            abstain_reason=None,
            decision_session=session,
            decision_at=decision_at,
            theme_components=components,
            stock_scores=pd.Series({code: .5 for code in members}, name='theme_score_equal_weight_v1'),
            coverage=dict(
                primary_endpoint='theme_score_equal_weight_v1',
                component_results_diagnostic_only=True,
                ledger_id='test-ledger',
                ledger_journal_hash=self.digest('ledger-journal'),
            ),
            receipt_hashes=(self.digest('receipt-1'),),
        )
        ordering = EntryOrdering(
            baseline_candidate_order=members,
            retained_holdings=(members[0],),
            new_entry_order=members[1:],
            applied=True,
            abstain_reason=None,
        )
        if pool_id == 'U0':
            return ThemeCarrierEntryArtifact(u0=universe, result=result, ordering=ordering)
        return ThemeCarrierEntryArtifact(universe=universe, result=result, ordering=ordering)

    def surface(self, pool_id='U0', identities=None, artifact=None):
        immutable_identities = {
            name: self.digest(f'U0:{name}')
            for name in FixedSurface._REQUIRED_IDENTITIES
        }
        if artifact is not None:
            immutable_identities.update(
                universe_definition=entry_universe_contract_sha256(pool_id),
                membership=artifact.universe.membership_hash,
                candidate_set=entry_only_instrument_set_sha256(artifact.ordering.baseline_candidate_order),
                retained_holdings=entry_only_instrument_set_sha256(artifact.ordering.retained_holdings),
            )
        if pool_id == 'U1' and artifact is None:
            for name in ('universe_definition', 'membership', 'candidate_set'):
                immutable_identities[name] = self.digest(f'U1:{name}')
        immutable_identities.update(identities or {})
        return FixedSurface(
            pool_contract=pool_id,
            fees_contract='fees-v1',
            holdings_contract='holdings-v1',
            risk_contract='risk-v1',
            retention_contract='retain-v1',
            cadence_contract='cadence-20-sessions-v1',
            execution_contract='next-open-v1',
            immutable_identities=immutable_identities,
        )

    def ledger_receipt(
        self,
        *,
        source_id,
        source_family,
        origin_id,
        raw,
        first='2026-09-30 17:10:00+08:00',
    ):
        # A test fixture can control the ledger clock; a real source caller
        # cannot establish eligibility by passing `now=`.
        self._ledger_now = first
        return self.ledger.record_receipt(
            source_id=source_id,
            source_family=source_family,
            origin_id=origin_id,
            canonical_url=f'https://example.test/{source_id}',
            raw=raw,
            content_type='application/json',
            published_at='2026-09-30 17:00:00+08:00',
            first_captured_at=first,
            available_at=first,
            availability_basis='first_capture',
            now=first,
        )

    def mirofish_coverage(self):
        decision = '2026-09-30 18:00:00+08:00'
        receipt = self.ledger_receipt(
            source_id='mirofish-text-source', source_family='mirofish-text-source',
            origin_id='mirofish-story', raw=b'mirofish text receipt',
        )
        self._ledger_now = '2026-09-30 17:20:00+08:00'
        entity = self.ledger.record_entity(
            receipt['event_id'], entity_type='stock', canonical_id='sz.000001',
            display_name='sz.000001', mapping_method='fixture', mapping_version='v1',
            now='2026-09-30 17:20:00+08:00',
        )
        self._ledger_now = '2026-09-30 17:50:00+08:00'
        self.ledger.declare_scope(
            'mirofish-u0', decision, ['sz.000001'], definition_version='u0-fixture-v1',
            source_hash=self.digest('mirofish-u0-members'), now='2026-09-30 17:50:00+08:00',
        )
        # This fixture needs an observed text seed; an 18:01 write is now an
        # explicit abstention by design, not an eligible simulation input.
        self._ledger_now = '2026-09-30 17:59:00+08:00'
        self.ledger.record_observation(
            observation_key='mirofish-text-coverage', scope_id='mirofish-u0', decision_at=decision,
            feature_id='text_consensus_v1', target_entity_keys=['sz.000001'],
            receipt_ids=[receipt['event_id']], entity_ids=[entity['event_id']], value=1.0,
            unit='count', extractor={'kind': 'rule', 'version': 'fixture-v1'},
            now='2026-09-30 18:01:00+08:00',
        )
        return MiroFishRunContract.ledger_coverage_evidence(
            ledger=self.ledger, scope_id='mirofish-u0', decision_at=decision,
            feature_id='text_consensus_v1',
        )

    def test_d2_must_be_archived_before_an_account_can_be_registered(self):
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_not_archived'):
            self.decision_gate(status='running').require_registration_evidence()
        gate=self.decision_gate()
        evidence = gate.require_registration_evidence()
        self.assertTrue(evidence['d2_archived'])
        self.assertFalse(evidence['account_execution_authorized'])
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_gate_registration_only'):
            gate.require_account_permission()
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_selection_basis_invalid'):
            self.decision_gate(selection_basis='highest_single_cell_annualized').require_registration_evidence()
        forged = self.decision_gate()
        forged_manifest = dict(forged.decision_manifest, selection_basis='highest_single_cell_annualized')
        forged = D2ArchiveGate(
            forged.run_id, forged.status, forged.protocol_sha256, forged.decision_sha256,
            forged_manifest, forged.archive_path,
        )
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_decision_manifest_hash_mismatch'):
            forged.require_registration_evidence()
        tampered_archive = self.decision_gate()
        Path(tampered_archive.archive_path).write_bytes(b'tampered archive')
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_archive_artifact_hash_mismatch'):
            tampered_archive.require_registration_evidence()

    def test_d2_archive_identity_cannot_be_rebound_to_another_run_after_rehashing(self):
        gate = self.decision_gate()
        archive = json.loads(Path(gate.archive_path).read_text())
        archive['run_id'] = 'other-run'
        identity_payload = {
            key: archive[key]
            for key in (
                'schema_version', 'run_id', 'protocol_sha256', 'selection_basis', 'full_matrix_sha256'
            )
        }
        archive['archive_identity_sha256'] = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        raw = json.dumps(archive, sort_keys=True, separators=(',', ':')).encode()
        Path(gate.archive_path).write_bytes(raw)
        manifest = dict(
            gate.decision_manifest,
            archive_sha256=hashlib.sha256(raw).hexdigest(),
            archive_identity_sha256=archive['archive_identity_sha256'],
        )
        decision_sha = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        forged = D2ArchiveGate(
            gate.run_id, gate.status, gate.protocol_sha256, decision_sha, manifest, gate.archive_path
        )
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_archive_artifact_identity_mismatch'):
            forged.require_registration_evidence()

    def test_ab_keeps_everything_except_the_verified_theme_carrier_entry_order_fixed(self):
        artifact = self.theme_artifact()
        baseline=dict(surface=self.surface(artifact=artifact), account_gate=None)
        treatment=dict(surface=self.surface(artifact=artifact), theme_carrier_entry_artifact=artifact, account_gate=None)
        result=validate_entry_sort_ab(baseline,treatment)
        self.assertEqual(result['changed_fields'],('theme_carrier_minimal_v1_entry_order',))
        self.assertEqual(result['u0_membership_sha256'], artifact.u0.membership_hash)
        bad=dict(treatment);bad['surface']=self.surface('U1', artifact=artifact)
        with self.assertRaisesRegex(ExternalResearchContractError, 'ab_surface_changed:pool_contract'):
            validate_entry_sort_ab(baseline,bad)

    def test_ab_rejects_string_sorter_and_tampered_theme_entry_artifact(self):
        artifact = self.theme_artifact()
        baseline = dict(surface=self.surface(artifact=artifact))
        treatment = dict(surface=self.surface(artifact=artifact), theme_carrier_entry_artifact=artifact)
        with self.assertRaisesRegex(ExternalResearchContractError, 'entry_sorter_label_forbidden'):
            validate_entry_sort_ab(dict(baseline, entry_sorter='reference'), treatment)
        malformed_order = EntryOrdering(
            baseline_candidate_order=artifact.ordering.baseline_candidate_order,
            retained_holdings=artifact.ordering.retained_holdings,
            new_entry_order=tuple(reversed(artifact.ordering.new_entry_order)),
            applied=True,
            abstain_reason=None,
        )
        malformed = ThemeCarrierEntryArtifact(
            u0=artifact.u0, result=artifact.result, ordering=malformed_order
        )
        with self.assertRaisesRegex(ExternalResearchContractError, 'entry_ordering_not_score_derived'):
            validate_entry_sort_ab(baseline, dict(treatment, theme_carrier_entry_artifact=malformed))

    def test_u1_is_a_replication_arm_not_an_u0_replacement(self):
        u0_artifact = self.theme_artifact()
        u1_artifact = self.theme_artifact(pool_id='U1')
        baseline=dict(surface=self.surface(artifact=u0_artifact),account_gate=None)
        treatment=dict(surface=self.surface('U1', artifact=u1_artifact),account_gate=None,
                       replication_of='U0',replace_u0=False,
                       u1_replication_protocol_sha256=self.digest('u1-protocol'),
                       theme_carrier_entry_artifact=u1_artifact,
                       u0_source_theme_carrier_entry_artifact=u0_artifact)
        result = validate_entry_sort_ab(baseline,treatment,allow_u1_replication=True)
        self.assertEqual(result['arm'],'U1')
        self.assertEqual(result['u1_membership_sha256'], u1_artifact.universe.membership_hash)
        self.assertEqual(
            result['u0_source_membership_sha256'], u0_artifact.universe.membership_hash
        )
        missing_u1 = dict(treatment)
        missing_u1.pop('theme_carrier_entry_artifact')
        with self.assertRaisesRegex(ExternalResearchContractError, 'theme_carrier_entry_artifact_required'):
            validate_entry_sort_ab(baseline, missing_u1, allow_u1_replication=True)
        forged_u0_source = dict(treatment, u0_source_theme_carrier_entry_artifact=u1_artifact)
        with self.assertRaisesRegex(ExternalResearchContractError, 'theme_carrier_verified_u0_required'):
            validate_entry_sort_ab(baseline, forged_u0_source, allow_u1_replication=True)
        treatment['replace_u0']=True
        with self.assertRaisesRegex(ExternalResearchContractError,'u1_cannot_replace_u0'):
            validate_entry_sort_ab(baseline,treatment,allow_u1_replication=True)

    def test_u0_immutable_identity_cannot_be_relabelled_from_a_broad_pool(self):
        artifact = self.theme_artifact()
        baseline=dict(surface=self.surface(artifact=artifact), account_gate=None)
        treatment=dict(
            surface=self.surface(artifact=artifact, identities={'membership': self.digest('u-broad-membership')}),
            theme_carrier_entry_artifact=artifact, account_gate=None,
        )
        with self.assertRaisesRegex(ExternalResearchContractError, 'ab_surface_changed:immutable_identities'):
            validate_entry_sort_ab(baseline, treatment)

    def test_account_registration_binds_archived_d2_not_just_the_ab_surface(self):
        artifact = self.theme_artifact()
        baseline = dict(surface=self.surface(artifact=artifact), account_gate=None)
        treatment = dict(surface=self.surface(artifact=artifact), theme_carrier_entry_artifact=artifact, account_gate=None)
        registration = register_atomic_entry_sort_experiment(
            experiment_id='theme-carrier-u0-v1',
            preregistration_sha256=self.digest('atomic-preregistration'),
            baseline=baseline,
            treatment=treatment,
            d2_gate=self.decision_gate(),
        )
        self.assertTrue(registration['registration_valid'])
        self.assertFalse(registration['account_execution_authorized'])
        with self.assertRaisesRegex(ExternalResearchContractError, 'd2_not_archived'):
            register_atomic_entry_sort_experiment(
                experiment_id='theme-carrier-u0-v1',
                preregistration_sha256=self.digest('atomic-preregistration'),
                baseline=baseline,
                treatment=treatment,
                d2_gate=self.decision_gate(status='running'),
            )

    def test_text_consensus_requires_two_independent_non_reprint_sources(self):
        decision='2026-09-30 18:00:00+08:00'
        first = self.ledger_receipt(
            source_id='publisher-a', source_family='publisher-a', origin_id='a:1', raw=b'first story'
        )
        second = self.ledger_receipt(
            source_id='publisher-b', source_family='publisher-b', origin_id='b:7', raw=b'second story',
            first='2026-09-30 17:30:00+08:00',
        )
        sources=[
            dict(receipt_id=first['event_id'], is_reprint=False),
            dict(receipt_id=second['event_id'], is_reprint=False),
        ]
        # Old hand-shaped maps cannot manufacture a consensus.
        self.assertEqual(TextConsensus.confirm(sources,decision)['reason'],'text_ledger_evidence_required')
        self.assertEqual(TextConsensus.confirm(sources,decision,ledger=self.ledger)['status'],'confirmed')
        sources[1]['source_family']='publisher-a'
        # Claimed source metadata is not trusted over the ledger.
        self.assertEqual(TextConsensus.confirm(sources,decision,ledger=self.ledger)['status'],'confirmed')
        sources[1]['is_reprint']=True
        self.assertEqual(TextConsensus.confirm(sources,decision,ledger=self.ledger)['status'],'abstain')

        # A caller cannot turn the same ledger-captured content into two
        # "independent" originals merely by supplying different source maps.
        copied = self.ledger_receipt(
            source_id='publisher-c', source_family='publisher-c', origin_id='c:9',
            raw=b'first story', first='2026-09-30 17:40:00+08:00',
        )
        forged_independence = [
            dict(receipt_id=first['event_id'], is_reprint=False, origin_id='forged-a'),
            dict(receipt_id=copied['event_id'], is_reprint=False, origin_id='forged-c'),
        ]
        self.assertEqual(
            TextConsensus.confirm(forged_independence, decision, ledger=self.ledger)['status'],
            'abstain',
        )

    def test_text_consensus_requires_the_fixed_1800_decision_cutoff(self):
        first = self.ledger_receipt(
            source_id='publisher-a', source_family='publisher-a', origin_id='a:1', raw=b'first story',
        )
        second = self.ledger_receipt(
            source_id='publisher-b', source_family='publisher-b', origin_id='b:7', raw=b'second story',
            first='2026-09-30 17:30:00+08:00',
        )
        result = TextConsensus.confirm(
            [dict(receipt_id=first['event_id']), dict(receipt_id=second['event_id'])],
            '2026-09-30 17:59:00+08:00',
            ledger=self.ledger,
        )
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['reason'], 'text_decision_cutoff_required')

    def test_text_consensus_rejects_a_malformed_reprint_marker(self):
        decision='2026-09-30 18:00:00+08:00'
        first = self.ledger_receipt(
            source_id='publisher-a', source_family='publisher-a', origin_id='a:1', raw=b'first story',
        )
        second = self.ledger_receipt(
            source_id='publisher-b', source_family='publisher-b', origin_id='b:7', raw=b'second story',
            first='2026-09-30 17:30:00+08:00',
        )
        result = TextConsensus.confirm(
            [
                dict(receipt_id=first['event_id'], is_reprint=False),
                dict(receipt_id=second['event_id'], is_reprint='false'),
            ],
            decision,
            ledger=self.ledger,
        )
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['reason'], 'text_reprint_status_invalid')

    def test_multi_agent_brief_cannot_rate_or_feed_unsettled_outcomes(self):
        receipt = self.ledger_receipt(
            source_id='agent-source', source_family='agent-source', origin_id='agent-story', raw=b'agent source'
        )
        brief=dict(instrument_id='sz.000001',facts=['f'],citations=[receipt['event_id']],catalysts=['c'],
                   invalidation_conditions=['i'],disagreements=['d'],abstain_reason=None,
                   outcome_feedback_ids=[],model_id='research-model-v1',prompt_sha256=self.digest('agent-prompt'),
                   decision_at='2026-09-30 18:00:00+08:00')
        receipts={receipt['event_id']: {'status':'accepted','raw_sha256':self.digest('receipt-1')}}
        with self.assertRaisesRegex(ExternalResearchContractError,'agent_citation_ledger_required'):
            validate_multi_agent_brief(brief,{'sz.000001'}, {},receipt_evidence=receipts)
        self.assertEqual(validate_multi_agent_brief(
            brief,{'sz.000001'}, {},ledger=self.ledger,receipt_evidence=receipts
        )['status'],'ok')
        with self.assertRaisesRegex(ExternalResearchContractError, 'agent_brief_decision_cutoff_required'):
            validate_multi_agent_brief(
                dict(brief, decision_at='2026-09-30 17:59:00+08:00'),
                {'sz.000001'}, {}, ledger=self.ledger,
            )
        rated=dict(brief,rating='buy')
        with self.assertRaisesRegex(ExternalResearchContractError,'agent_direct_rating_forbidden'):
            validate_multi_agent_brief(rated,{'sz.000001'}, {},ledger=self.ledger,receipt_evidence=receipts)
        nested=dict(brief,facts=[{'trade':'buy'}])
        with self.assertRaisesRegex(ExternalResearchContractError,'agent_direct_rating_forbidden'):
            validate_multi_agent_brief(nested,{'sz.000001'}, {},ledger=self.ledger,receipt_evidence=receipts)
        hidden_order=dict(brief,facts=[{'entry_order':['sz.000001']}])
        with self.assertRaisesRegex(ExternalResearchContractError,'agent_direct_rating_forbidden'):
            validate_multi_agent_brief(hidden_order,{'sz.000001'}, {},ledger=self.ledger)
        hidden_feedback=dict(brief,facts=[{'outcome_feedback_ids':['pending']}])
        with self.assertRaisesRegex(ExternalResearchContractError,'agent_unregistered_outcome_feedback_forbidden'):
            validate_multi_agent_brief(hidden_feedback,{'sz.000001'}, {},ledger=self.ledger)
        fed=dict(brief,outcome_feedback_ids=['pending'])
        with self.assertRaisesRegex(ExternalResearchContractError,'unsettled_outcome_feedback'):
            validate_multi_agent_brief(
                fed,{'sz.000001'},{'pending':'pending'},ledger=self.ledger,receipt_evidence=receipts
            )
        # A hand-shaped status map cannot turn an unrecorded outcome into
        # settled feedback.
        forged_settlement = dict(brief, outcome_feedback_ids=['invented-outcome'])
        with self.assertRaisesRegex(ExternalResearchContractError, 'unsettled_outcome_feedback'):
            validate_multi_agent_brief(
                forged_settlement,
                {'sz.000001'}, {'invented-outcome': 'settled'}, ledger=self.ledger,
            )

        # A genuine ledger outcome is usable only after both its settlement
        # record and its declared availability are on or before this brief.
        self._ledger_now = '2026-09-30 17:50:00+08:00'
        self.ledger.declare_scope(
            'agent-outcome-scope', '2026-09-30 18:00:00+08:00', ['sz.000001'],
            definition_version='fixture-v1', source_hash=self.digest('agent-outcome-scope'),
            now='2026-09-30 17:50:00+08:00',
        )
        entity = self.ledger.record_entity(
            receipt['event_id'], entity_type='stock', canonical_id='sz.000001',
            display_name='sz.000001', mapping_method='fixture', mapping_version='v1',
            now='2026-09-30 17:20:00+08:00',
        )
        self._ledger_now = '2026-09-30 17:59:00+08:00'
        observation = self.ledger.record_observation(
            observation_key='agent-outcome-observation', scope_id='agent-outcome-scope',
            decision_at='2026-09-30 18:00:00+08:00', feature_id='agent-fact-v1',
            target_entity_keys=['sz.000001'], receipt_ids=[receipt['event_id']],
            entity_ids=[entity['event_id']], value=1.0, unit='count',
            extractor={'kind': 'rule', 'version': 'fixture-v1'},
            now='2026-09-30 18:01:00+08:00',
        )
        outcome = self.ledger.record_outcome(
            observation_id=observation['event_id'], outcome_key='agent-outcome-v1',
            label_definition={'version': 'fixture-v1'},
            entry_at='2026-10-01 09:30:00+08:00', exit_at='2026-10-02 09:30:00+08:00',
            available_at='2026-10-02 18:00:00+08:00', value=.01,
            receipt_ids=[receipt['event_id']], now='2026-10-02 18:00:00+08:00',
        )
        future_settlement = dict(brief, outcome_feedback_ids=[outcome['event_id']])
        with self.assertRaisesRegex(ExternalResearchContractError, 'unsettled_outcome_feedback'):
            validate_multi_agent_brief(
                future_settlement, {'sz.000001'}, {outcome['event_id']: 'settled'}, ledger=self.ledger,
            )
        settled_result = validate_multi_agent_brief(
            dict(future_settlement, decision_at='2026-10-02 18:00:00+08:00'),
            {'sz.000001'}, {}, ledger=self.ledger,
        )
        self.assertEqual(settled_result['feedback_outcomes'], (outcome['event_id'],))
        with self.assertRaisesRegex(ExternalResearchContractError,'agent_citation_receipt_unverified'):
            validate_multi_agent_brief(
                dict(brief,citations=['unknown-receipt']), {'sz.000001'}, {}, ledger=self.ledger
            )

    def test_multi_agent_abstention_still_records_its_model_prompt_and_reason(self):
        abstention=dict(
            instrument_id='sz.000001', decision_at='2026-09-30 18:00:00+08:00',
            abstain_reason='insufficient independent evidence',
            model_id='research-model-v1', prompt_sha256=self.digest('agent-abstain-prompt'),
            outcome_feedback_ids=[],
        )
        result = validate_multi_agent_brief(abstention, {'sz.000001'}, {}, ledger=self.ledger)
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['feedback_outcomes'], ())
        with self.assertRaisesRegex(ExternalResearchContractError, 'agent_model_id_required'):
            validate_multi_agent_brief(
                dict(abstention, model_id=None), {'sz.000001'}, {}, ledger=self.ledger,
            )
        with self.assertRaisesRegex(ExternalResearchContractError, 'agent_abstention_reason_required'):
            validate_multi_agent_brief(
                dict(abstention, abstain_reason={'why': 'missing'}), {'sz.000001'}, {}, ledger=self.ledger,
            )

    def test_mirofish_requires_coverage_k8_real_logs_and_ignores_unused_parameters(self):
        coverage=self.mirofish_coverage()
        runs=[]
        for i in range(8):
            actions=[dict(actor_id='a',action='post'),dict(actor_id=f'b{i}',action='share')]
            runs.append(dict(
                run_id=str(i),seed=i,model_id='simulation-model-v1',prompt_sha256=self.digest('simulation-prompt'),
                action_log_hash=MiroFishRunContract.action_log_sha256(rounds=2,actions=actions),rounds=2,
                actions=actions,
            ))
        with self.assertRaisesRegex(ExternalResearchContractError, 'mirofish_ledger_coverage_required'):
            MiroFishRunContract.summarize(coverage=coverage,minimum_coverage=.75,runs=runs)
        result=MiroFishRunContract.summarize(
            coverage=coverage,minimum_coverage=.75,runs=runs,ledger=self.ledger
        )
        self.assertEqual(result['runs'],8)
        self.assertIn('iqr',result)
        forged_coverage = dict(coverage, eligible_rate=.8)
        with self.assertRaisesRegex(ExternalResearchContractError, 'mirofish_ledger_coverage_mismatch:eligible_rate'):
            MiroFishRunContract.summarize(
                coverage=forged_coverage, minimum_coverage=.75, runs=runs, ledger=self.ledger
            )
        runs[0]['stance']='bullish'
        with self.assertRaisesRegex(ExternalResearchContractError,'mirofish_unused_parameter_forbidden'):
            MiroFishRunContract.summarize(coverage=coverage,minimum_coverage=.75,runs=runs,ledger=self.ledger)
        runs[0].pop('stance')
        runs[0]['metadata']={'sentiment_bias':'bullish'}
        with self.assertRaisesRegex(ExternalResearchContractError,'mirofish_unused_parameter_forbidden'):
            MiroFishRunContract.summarize(coverage=coverage,minimum_coverage=.75,runs=runs,ledger=self.ledger)
        runs[0].pop('metadata')
        runs[0]['actions'].append(dict(actor_id='c',action='reply'))
        with self.assertRaisesRegex(ExternalResearchContractError,'mirofish_action_log_hash_mismatch'):
            MiroFishRunContract.summarize(coverage=coverage,minimum_coverage=.75,runs=runs,ledger=self.ledger)

    def test_action_log_factors_reject_stance_without_a_simulation(self):
        factors = MiroFishRunContract.factors_from_action_log(
            rounds=2,
            actions=(
                dict(actor_id="a", action="post"),
                dict(actor_id="b", action="share"),
                dict(actor_id="a", action="reply"),
            ),
        )
        self.assertEqual(factors["propagation_range"], 2)
        self.assertEqual(factors["action_density"], 1.5)
        self.assertEqual(factors["simulation_runs"], 0)
        self.assertFalse(factors["account_result"])
        with self.assertRaisesRegex(ExternalResearchContractError, "mirofish_unused_parameter_forbidden"):
            MiroFishRunContract.factors_from_action_log(
                rounds=1, actions=(dict(actor_id="a", action="post", stance="bullish"),)
            )
        with self.assertRaisesRegex(ExternalResearchContractError, "mirofish_unused_parameter_forbidden"):
            MiroFishRunContract.factors_from_action_log(
                rounds=1, actions=(dict(actor_id="a", action="post", sentiment_bias=1),)
            )

    def test_brief_sections_require_ledger_citations_and_reject_orders(self):
        receipt = self.ledger_receipt(
            source_id="brief-source", source_family="brief-source", origin_id="brief-story", raw=b"brief source"
        )
        brief = dict(
            instrument_id="sz.000001",
            facts=["observed fact"],
            citations=[receipt["event_id"]],
            catalysts=["named catalyst"],
            invalidation_conditions=["named failure"],
            disagreements=["named disagreement"],
            model_id="research-model-v1",
            prompt_sha256=self.digest("brief-prompt"),
            decision_at="2026-09-30 18:00:00+08:00",
            outcome_feedback_ids=[],
        )
        accepted = validate_multi_agent_brief(brief, {"sz.000001"}, {}, ledger=self.ledger)
        self.assertEqual(accepted["status"], "ok")
        self.assertEqual(accepted["direct_ordering_allowed"], False)
        self.assertEqual(accepted["citation_receipt_ids"], (receipt["event_id"],))
        with self.assertRaisesRegex(ExternalResearchContractError, "agent_brief_missing:invalidation_conditions"):
            validate_multi_agent_brief(
                {key: value for key, value in brief.items() if key != "invalidation_conditions"},
                {"sz.000001"}, {}, ledger=self.ledger,
            )
        with self.assertRaisesRegex(ExternalResearchContractError, "agent_direct_rating_forbidden"):
            validate_multi_agent_brief(dict(brief, order={"side": "buy"}), {"sz.000001"}, {}, ledger=self.ledger)

    def test_mainline_matrix_uses_full_phase_cost_matrix_not_best_cell(self):
        cells=[]
        for phase in (0,4,8,12,16):
            for cost in ('base','double'):
                cells.append(dict(phase=phase,cost=cost,after_fee_annualized=.10+phase/1000,
                                  max_drawdown=.24))
        result=MainlineMatrix.evaluate(cells)
        self.assertEqual(result['selection_metric'],'median_after_fee_annualized')
        self.assertIn('highest_cell_disclosure',result)
        cells[-1]['max_drawdown']=.251
        with self.assertRaisesRegex(ExternalResearchContractError,'mainline_drawdown_exceeds_25pct'):
            MainlineMatrix.evaluate(cells)
        cells[-1]['max_drawdown']=-.251
        with self.assertRaisesRegex(ExternalResearchContractError,'mainline_drawdown_exceeds_25pct'):
            MainlineMatrix.evaluate(cells)

    def test_fusion_and_structure_wait_for_a_fixed_single_expert_increment(self):
        with self.assertRaisesRegex(ExternalResearchContractError,'single_expert_increment_missing'):
            validate_fusion_or_structure('fusion', [{'expert_id':'theme','pre_registered_net_increment':False}],
                                         signal_fixed=True)
        with self.assertRaisesRegex(ExternalResearchContractError,'signal_not_fixed'):
            validate_fusion_or_structure('structure', [{'expert_id':'theme','pre_registered_net_increment':True}],
                                         signal_fixed=False)
        self.assertTrue(validate_fusion_or_structure('structure',
            [{'expert_id':'theme','pre_registered_net_increment':True}],signal_fixed=True)['allowed'])

    def test_disclosure_must_include_search_family_failures_coverage_and_identities(self):
        complete=dict(search_family=['theme-v1'],all_outcomes=[{'status':'failed'}],coverage={'eligible_rate':.7},
                      code_identity='c'*64,data_identity='d'*64,model_identity='m-v1',
                      decision_impact='keep_research_only')
        self.assertTrue(validate_research_disclosure(complete)['complete'])
        incomplete=dict(complete);del incomplete['all_outcomes']
        with self.assertRaisesRegex(ExternalResearchContractError,'disclosure_missing:all_outcomes'):
            validate_research_disclosure(incomplete)


if __name__ == '__main__':
    unittest.main()
