/**
 * @file xhci_arch_sim.cpp
 * @brief Chapter 6.2: USB 3.0 & xHCI Architecture Simulation (MikanOS xHCI Driver Theory)
 *
 * Simulates xHCI 16-byte TRB bitfields, circular Ring buffer with Link TRB and Cycle Bit toggle,
 * 32-byte Slot/Endpoint Contexts, and lock-free producer-consumer synchronization.
 */

#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <string>
#include <memory>

// =============================================================================
// 1. TRB (Transfer Request Block) Definitions (xHCI Spec 1.2 Section 4.11 / 6.4)
// =============================================================================

namespace usb::xhci {

enum class TRBType : uint32_t {
    kReserved = 0,
    kNormal = 1,
    kSetupStage = 2,
    kDataStage = 3,
    kStatusStage = 4,
    kIsoch = 5,
    kLink = 6,
    kEventData = 7,
    kNoOp = 8,
    kEnableSlotCommand = 9,
    kDisableSlotCommand = 10,
    kAddressDeviceCommand = 11,
    kConfigureEndpointCommand = 12,
    kEvaluateContextCommand = 13,
    kResetEndpointCommand = 14,
    kStopEndpointCommand = 15,
    kSetTRDequeuePointerCommand = 16,
    kResetDeviceCommand = 17,
    kTransferEvent = 32,
    kCommandCompletionEvent = 33,
    kPortStatusChangeEvent = 34,
    kBandwidthRequestEvent = 35,
    kDoorbellEvent = 36,
    kHostControllerEvent = 37,
    kDeviceNotificationEvent = 38,
    kMFINDEXWrapEvent = 39,
};

enum class CompletionCode : uint32_t {
    kInvalid = 0,
    kSuccess = 1,
    kDataBufferError = 2,
    kBabbleDetectedError = 3,
    kUSBTransactionError = 4,
    kTRBError = 5,
    kStallError = 6,
    kResourceError = 7,
    kBandwidthError = 8,
    kNoSlotsAvailableError = 9,
    kInvalidStreamTypeError = 10,
    kSlotNotEnabledError = 11,
    kEndpointNotEnabledError = 12,
    kShortPacket = 13,
    kRingUnderrun = 14,
    kRingOverrun = 15,
    kVFEventRingFullError = 16,
    kParameterError = 17,
    kBandwidthOverrunError = 18,
    kContextStateError = 19,
    kNoPingResponseError = 20,
    kEventRingFullError = 21,
    kIncompatibleDeviceError = 22,
    kMissedServiceError = 23,
    kCommandRingStopped = 24,
    kCommandAborted = 25,
    kStopped = 26,
    kStoppedLengthInvalid = 27,
    kControlAbortError = 28,
};

/** @brief 16-byte fixed-size Transfer Request Block (TRB) */
union TRB {
    std::array<uint32_t, 4> data{};
    struct {
        uint64_t parameter;
        uint32_t status;
        uint32_t cycle_bit : 1;
        uint32_t evaluate_next_trb : 1;
        uint32_t : 8;
        uint32_t trb_type : 6;
        uint32_t control : 16;
    } __attribute__((packed)) bits;
};
static_assert(sizeof(TRB) == 16, "TRB must be exactly 16 bytes");

/** @brief Normal TRB for bulk/interrupt/isoch data transfers */
union NormalTRB {
    static const unsigned int Type = 1;
    std::array<uint32_t, 4> data{};
    struct {
        uint64_t data_buffer_pointer;

        uint32_t trb_transfer_length : 17;
        uint32_t td_size : 5;
        uint32_t interrupter_target : 10;

        uint32_t cycle_bit : 1;
        uint32_t evaluate_next_trb : 1;
        uint32_t interrupt_on_short_packet : 1;
        uint32_t no_snoop : 1;
        uint32_t chain_bit : 1;
        uint32_t interrupt_on_completion : 1;
        uint32_t immediate_data : 1;
        uint32_t : 2;
        uint32_t block_event_interrupt : 1;
        uint32_t trb_type : 6;
        uint32_t : 16;
    } __attribute__((packed)) bits;

    NormalTRB() {
        bits.trb_type = Type;
    }
};
static_assert(sizeof(NormalTRB) == 16, "NormalTRB must be exactly 16 bytes");

/** @brief Link TRB for chaining ring segments and toggling cycle state */
union LinkTRB {
    static const unsigned int Type = 6;
    std::array<uint32_t, 4> data{};
    struct {
        uint64_t ring_segment_pointer;

        uint32_t : 22;
        uint32_t interrupter_target : 10;

        uint32_t cycle_bit : 1;
        uint32_t toggle_cycle : 1;
        uint32_t : 8;
        uint32_t trb_type : 6;
        uint32_t : 16;
    } __attribute__((packed)) bits;

    LinkTRB(uint64_t ring_ptr) {
        bits.trb_type = Type;
        bits.ring_segment_pointer = ring_ptr;
    }
};
static_assert(sizeof(LinkTRB) == 16, "LinkTRB must be exactly 16 bytes");

/** @brief Enable Slot Command TRB */
union EnableSlotCommandTRB {
    static const unsigned int Type = 9;
    std::array<uint32_t, 4> data{};
    struct {
        uint32_t : 32;
        uint32_t : 32;
        uint32_t : 32;
        uint32_t cycle_bit : 1;
        uint32_t : 9;
        uint32_t trb_type : 6;
        uint32_t slot_type : 5;
        uint32_t : 11;
    } __attribute__((packed)) bits;

    EnableSlotCommandTRB() {
        bits.trb_type = Type;
    }
};
static_assert(sizeof(EnableSlotCommandTRB) == 16, "EnableSlotCommandTRB must be 16 bytes");

/** @brief Command Completion Event TRB */
union CommandCompletionEventTRB {
    static const unsigned int Type = 33;
    std::array<uint32_t, 4> data{};
    struct {
        uint64_t command_trb_pointer;

        uint32_t command_completion_parameter : 24;
        uint32_t completion_code : 8;

        uint32_t cycle_bit : 1;
        uint32_t : 9;
        uint32_t trb_type : 6;
        uint32_t vf_id : 8;
        uint32_t slot_id : 8;
    } __attribute__((packed)) bits;

    CommandCompletionEventTRB() {
        bits.trb_type = Type;
    }
};
static_assert(sizeof(CommandCompletionEventTRB) == 16, "CommandCompletionEventTRB must be 16 bytes");

// =============================================================================
// 2. Context Data Structures (xHCI Spec 1.2 Section 6.2)
// =============================================================================

/** @brief 32-byte Slot Context */
union SlotContext {
    uint32_t dwords[8]{};
    struct {
        uint32_t route_string : 20;
        uint32_t speed : 4;
        uint32_t : 1;
        uint32_t mtt : 1;
        uint32_t hub : 1;
        uint32_t context_entries : 5;

        uint32_t max_exit_latency : 16;
        uint32_t root_hub_port_num : 8;
        uint32_t num_ports : 8;

        uint32_t tt_hub_slot_id : 8;
        uint32_t tt_port_num : 8;
        uint32_t ttt : 2;
        uint32_t : 4;
        uint32_t interrupter_target : 10;

        uint32_t usb_device_address : 8;
        uint32_t : 19;
        uint32_t slot_state : 5;

        uint32_t reserved[4];
    } __attribute__((packed)) bits;
};
static_assert(sizeof(SlotContext) == 32, "SlotContext must be exactly 32 bytes");

/** @brief 32-byte Endpoint Context */
union EndpointContext {
    uint32_t dwords[8]{};
    struct {
        uint32_t ep_state : 3;
        uint32_t : 5;
        uint32_t mult : 2;
        uint32_t max_primary_streams : 5;
        uint32_t linear_stream_array : 1;
        uint32_t interval : 8;
        uint32_t max_esit_payload_hi : 8;

        uint32_t : 1;
        uint32_t error_count : 2;
        uint32_t ep_type : 3;
        uint32_t : 1;
        uint32_t host_initiate_disable : 1;
        uint32_t max_burst_size : 8;
        uint32_t max_packet_size : 16;

        uint32_t dequeue_cycle_state : 1;
        uint32_t : 3;
        uint64_t tr_dequeue_pointer : 60;

        uint32_t average_trb_length : 16;
        uint32_t max_esit_payload_lo : 16;

        uint32_t reserved[3];
    } __attribute__((packed)) bits;
};
static_assert(sizeof(EndpointContext) == 32, "EndpointContext must be exactly 32 bytes");

/** @brief 1024-byte Device Context (SlotContext + 31 EndpointContexts) */
struct DeviceContext {
    SlotContext slot;
    EndpointContext ep[31];
};
static_assert(sizeof(DeviceContext) == 1024, "DeviceContext must be exactly 1024 bytes");

// =============================================================================
// 3. Ring Circular Buffer Implementation with Cycle Bit Handshake
// =============================================================================

class Ring {
public:
    Ring() = default;

    void Initialize(size_t buf_size, uint64_t ring_phys_addr = 0x1000) {
        buf_size_ = buf_size;
        ring_phys_addr_ = ring_phys_addr;
        buf_.assign(buf_size_, TRB{});
        cycle_bit_ = true;
        write_index_ = 0;
    }

    TRB* Push(const std::array<uint32_t, 4>& data) {
        auto trb_ptr = &buf_[write_index_];
        CopyToCurrent(data);

        ++write_index_;
        if (write_index_ == buf_size_ - 1) {
            // Write Link TRB at the end of the segment
            LinkTRB link{ring_phys_addr_};
            link.bits.toggle_cycle = 1;
            CopyToCurrent(link.data);

            write_index_ = 0;
            cycle_bit_ = !cycle_bit_; // Toggle Producer Cycle State (PCS)
        }
        return trb_ptr;
    }

    template <typename T>
    TRB* Push(const T& trb) {
        return Push(trb.data);
    }

    size_t BufferSize() const { return buf_size_; }
    size_t WriteIndex() const { return write_index_; }
    bool CycleBit() const { return cycle_bit_; }
    const TRB& At(size_t i) const { return buf_.at(i); }

private:
    std::vector<TRB> buf_{};
    size_t buf_size_ = 0;
    uint64_t ring_phys_addr_ = 0;
    bool cycle_bit_ = true;
    size_t write_index_ = 0;

    void CopyToCurrent(const std::array<uint32_t, 4>& data) {
        for (int i = 0; i < 3; ++i) {
            buf_[write_index_].data[i] = data[i];
        }
        // Force cycle bit in dword 3 to match Producer Cycle State
        buf_[write_index_].data[3]
            = (data[3] & 0xFFFFFFFEu) | static_cast<uint32_t>(cycle_bit_);
    }
};

// =============================================================================
// 4. xHCI Controller Architecture Simulator
// =============================================================================

class xHCIControllerSimulator {
public:
    xHCIControllerSimulator(size_t cmd_ring_size = 8, size_t max_slots = 4)
        : max_slots_{max_slots} {
        cmd_ring_.Initialize(cmd_ring_size, 0x2000);
        dcbaa_.resize(max_slots + 1, nullptr);
        event_ring_.resize(16);
    }

    Ring& CommandRing() { return cmd_ring_; }

    void RingDoorbell(uint8_t slot_id, uint8_t target) {
        if (slot_id == 0 && target == 0) {
            ProcessCommandRing();
        }
    }

    size_t EventCount() const { return event_count_; }
    const CommandCompletionEventTRB& LastEvent() const { return last_event_; }
    const DeviceContext* GetDeviceContext(uint8_t slot) const { return dcbaa_.at(slot); }

private:
    size_t max_slots_;
    Ring cmd_ring_;
    std::vector<DeviceContext*> dcbaa_;
    std::vector<DeviceContext> allocated_contexts_{};
    std::vector<TRB> event_ring_;
    size_t event_count_ = 0;
    CommandCompletionEventTRB last_event_{};
    size_t read_index_ = 0;
    bool consumer_cycle_state_ = true;

    void ProcessCommandRing() {
        while (true) {
            TRB trb = cmd_ring_.At(read_index_);
            // If Cycle Bit does not match consumer cycle state, no new TRBs!
            if (trb.bits.cycle_bit != consumer_cycle_state_) {
                break;
            }

            if (trb.bits.trb_type == LinkTRB::Type) {
                // Link TRB encountered!
                read_index_ = 0;
                consumer_cycle_state_ = !consumer_cycle_state_;
                continue;
            }

            if (trb.bits.trb_type == EnableSlotCommandTRB::Type) {
                uint8_t slot_id = 1;
                allocated_contexts_.push_back(DeviceContext{});
                dcbaa_[slot_id] = &allocated_contexts_.back();
                dcbaa_[slot_id]->slot.bits.slot_state = 1; // Default state
                dcbaa_[slot_id]->slot.bits.speed = 4;      // SuperSpeed

                // Emit event
                last_event_.bits.completion_code = static_cast<uint32_t>(CompletionCode::kSuccess);
                last_event_.bits.slot_id = slot_id;
                last_event_.bits.cycle_bit = 1;
                event_count_++;
            }

            read_index_ = (read_index_ + 1) % cmd_ring_.BufferSize();
        }
    }
};

} // namespace usb::xhci

// =============================================================================
// 5. Standalone Verification Suite
// =============================================================================

void RunTests() {
    using namespace usb::xhci;

    std::cout << "[TEST] 1. xHCI TRB & Context Memory Layout Verification..." << std::endl;
    assert(sizeof(TRB) == 16);
    assert(sizeof(NormalTRB) == 16);
    assert(sizeof(LinkTRB) == 16);
    assert(sizeof(EnableSlotCommandTRB) == 16);
    assert(sizeof(CommandCompletionEventTRB) == 16);
    assert(sizeof(SlotContext) == 32);
    assert(sizeof(EndpointContext) == 32);
    assert(sizeof(DeviceContext) == 1024);

    std::cout << "[TEST] 2. Bitfield Alignment & Offset Verification..." << std::endl;
    NormalTRB normal{};
    normal.bits.data_buffer_pointer = 0xDEADBEEF12345678ULL;
    normal.bits.trb_transfer_length = 4096;
    normal.bits.cycle_bit = 1;
    normal.bits.interrupt_on_completion = 1;

    TRB* raw_trb = reinterpret_cast<TRB*>(&normal);
    assert(raw_trb->bits.parameter == 0xDEADBEEF12345678ULL);
    assert(raw_trb->bits.trb_type == 1);
    assert(raw_trb->bits.cycle_bit == 1);

    std::cout << "[TEST] 3. Ring Push & Link TRB Cycle Bit Toggle Verification..." << std::endl;
    Ring ring;
    const size_t ring_size = 4; // 3 command slots + 1 Link TRB slot
    ring.Initialize(ring_size, 0x5000);

    assert(ring.CycleBit() == true);
    assert(ring.WriteIndex() == 0);

    EnableSlotCommandTRB cmd1{};
    ring.Push(cmd1);
    assert(ring.WriteIndex() == 1);
    assert(ring.CycleBit() == true);
    assert(ring.At(0).bits.cycle_bit == 1);
    assert(ring.At(0).bits.trb_type == EnableSlotCommandTRB::Type);

    EnableSlotCommandTRB cmd2{};
    ring.Push(cmd2);
    assert(ring.WriteIndex() == 2);
    assert(ring.CycleBit() == true);
    assert(ring.At(1).bits.cycle_bit == 1);

    // This push should hit index 2 (size - 2). Next write index would be 3 (size - 1),
    // which triggers LinkTRB injection and toggles CycleBit to false!
    EnableSlotCommandTRB cmd3{};
    ring.Push(cmd3);
    assert(ring.At(2).bits.cycle_bit == 1);
    // Link TRB must be at index 3
    assert(ring.At(3).bits.trb_type == LinkTRB::Type);
    assert(ring.At(3).bits.cycle_bit == 1); // Link TRB written during cycle_bit=true
    // Toggle cycle must wrap write_index to 0 and invert cycle_bit to false!
    assert(ring.WriteIndex() == 0);
    assert(ring.CycleBit() == false);

    std::cout << "[TEST] 4. Second Lap Inverted Cycle Bit Verification..." << std::endl;
    // Push on lap 2 (cycle_bit_ == false)
    ring.Push(cmd1);
    assert(ring.WriteIndex() == 1);
    assert(ring.CycleBit() == false);
    assert(ring.At(0).bits.cycle_bit == 0); // Must be 0 in lap 2!

    ring.Push(cmd2);
    assert(ring.WriteIndex() == 2);
    assert(ring.CycleBit() == false);
    assert(ring.At(1).bits.cycle_bit == 0);

    ring.Push(cmd3);
    assert(ring.At(2).bits.cycle_bit == 0);
    assert(ring.At(3).bits.trb_type == LinkTRB::Type);
    assert(ring.At(3).bits.cycle_bit == 0); // Written with cycle_bit=false
    // Wrapped back to 0 and cycle bit inverted back to true!
    assert(ring.WriteIndex() == 0);
    assert(ring.CycleBit() == true);

    std::cout << "[TEST] 5. Full Controller Doorbell & Event Ring Pipeline..." << std::endl;
    xHCIControllerSimulator controller(4, 2);
    EnableSlotCommandTRB enable_cmd{};
    controller.CommandRing().Push(enable_cmd);

    // Ring doorbell for command ring
    controller.RingDoorbell(0, 0);
    assert(controller.EventCount() == 1);
    assert(controller.LastEvent().bits.slot_id == 1);
    assert(controller.LastEvent().bits.completion_code == static_cast<uint32_t>(CompletionCode::kSuccess));
    assert(controller.GetDeviceContext(1) != nullptr);
    assert(controller.GetDeviceContext(1)->slot.bits.slot_state == 1);

    std::cout << "[SUCCESS] All Chapter 6.2 xHCI architecture simulation tests PASSED!" << std::endl;
}

int main(int argc, char* argv[]) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    if (test_mode) {
        RunTests();
        return 0;
    }

    std::cout << "=== MikanOS Chapter 6.2: USB 3.0 & xHCI Architecture Simulator ===" << std::endl;
    std::cout << "TRB Size: " << sizeof(usb::xhci::TRB) << " bytes" << std::endl;
    std::cout << "Slot Context Size: " << sizeof(usb::xhci::SlotContext) << " bytes" << std::endl;
    std::cout << "Endpoint Context Size: " << sizeof(usb::xhci::EndpointContext) << " bytes" << std::endl;
    std::cout << "Device Context Size: " << sizeof(usb::xhci::DeviceContext) << " bytes (1024 B)" << std::endl;
    std::cout << "Pass --test to execute verification suite." << std::endl;
    return 0;
}
